import paddle
from transformer_pytorch import FastTransformer
import numpy as np
import os
print(os.getcwd())
os.chdir("/root/baidu/paddle/transformer_pytorch/fake_data")
model = FastTransformer(num_tokens=20000, dim=512, depth=2, max_seq_len=\
    4096, absolute_pos_emb=True)
x_np = np.load('x.npy')
x = paddle.to_tensor(x_np)
mask_np= np.load('mask.npy')
mask = paddle.to_tensor(mask_np)
logits = model(x, mask=mask)
print(logits)
