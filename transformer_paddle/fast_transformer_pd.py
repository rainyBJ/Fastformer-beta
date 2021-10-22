import paddle
import paddle.nn.functional as F
from paddle import nn
from paddlenlp.ops import einsum 
import numpy as np
from einops import rearrange
from einops import reduce
from rotary_embedding_torch import apply_rotary_emb
from rotary_embedding_torch import RotaryEmbedding


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Layer):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


def FeedForward(dim, mult=4):
    return nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(
        dim * mult, dim))


class FastAttention(nn.Layer):

    def __init__(self, dim, *, heads=8, dim_head=64, max_seq_len=None,
        pos_emb=None):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias_attr=False)
        assert not (exists(pos_emb) and not exists(max_seq_len)
            ), 'max_seq_len must be passed in if to use rotary positional embeddings'
        self.pos_emb = pos_emb
        self.max_seq_len = max_seq_len
        kv_attn_proj_divisor = 1 if not exists(pos_emb) else 2
        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias_attr=False)
        self.to_k_attn_logits = nn.Linear(dim_head // kv_attn_proj_divisor,
            1, bias_attr=False)
        self.to_r = nn.Linear(dim_head // kv_attn_proj_divisor, dim_head)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask=None):
        n, device, h, use_rotary_emb = x.shape[1
            ], x.place, self.heads, exists(self.pos_emb)
        x = self.to_qkv(x)
        qkv = x.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        mask_value = -np.finfo('float32').max
        mask = rearrange(mask, 'b n -> b () n')
        if use_rotary_emb:
            freqs = self.pos_emb(paddle.arange(self.max_seq_len).
                requires_grad_(False), cache_key=self.max_seq_len)
            freqs = rearrange(freqs[:n], 'n d -> () () n d')
            q_aggr, k_aggr, v_aggr = map(lambda t: apply_rotary_emb(freqs,
                t), (q, k, v))
        else:
            q_aggr, k_aggr, v_aggr = q, k, v
        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n'
            ) * self.scale
        q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        q_attn = q_attn_logits.softmax(dim=-1)
        global_q = einsum('b h n, b h n d -> b h d', q_attn, q_aggr)
        global_q = rearrange(global_q, 'b h d -> b h () d')
        k = k * global_q
        if use_rotary_emb:
            k = reduce(k, 'b h n (d r) -> b h n d', 'sum', r=2)
        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n'
            ) * self.scale
        k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_attn = k_attn_logits.softmax(dim=-1)
        global_k = einsum('b h n, b h n d -> b h d', k_attn, k_aggr)
        global_k = rearrange(global_k, 'b h d -> b h () d')
        u = v_aggr * global_k
        if use_rotary_emb:
            u = reduce(u, 'b h n (d r) -> b h n d', 'sum', r=2)
        r = self.to_r(u)
        r = r + q
        r = rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)


class FastTransformer(nn.Layer):

    def __init__(self, *, num_tokens, dim, depth, max_seq_len, heads=8,
        dim_head=64, ff_mult=4, absolute_pos_emb=False):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim
            ) if absolute_pos_emb else None
        layer_pos_emb = None
        if not absolute_pos_emb:
            assert dim_head % 4 == 0, 'dimension of the head must be divisible by 4 to use rotary embeddings'
            layer_pos_emb = RotaryEmbedding(dim_head // 2)
        self.layers = nn.LayerList([])
        for _ in range(depth):
            attn = FastAttention(dim, dim_head=dim_head, heads=heads,
                pos_emb=layer_pos_emb, max_seq_len=max_seq_len)
            ff = FeedForward(dim, mult=ff_mult)
            self.layers.append(nn.LayerList([PreNorm(dim, attn), PreNorm(
                dim, ff)]))
        first_block, _ = self.layers[0]
        for block, _ in self.layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits
        self.to_logits = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,
            num_tokens))

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.place
        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            # pos_emb = self.abs_pos_emb(paddle.arange(n).requires_grad_(False))
            pos_emb = self.abs_pos_emb(paddle.arange(n))
            pos_emb_np = pos_emb.numpy()
            x = x + paddle.to_tensor(rearrange(pos_emb_np, 'n d -> () n d'))
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return self.to_logits(x)
