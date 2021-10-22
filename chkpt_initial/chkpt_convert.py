import paddle
import torch

chkpt_path = 'torch_init.pth'
paddle_chkpt_path = "paddle_init.pdparams"

paddle_dict = {}
torch_checkpoint = torch.load(chkpt_path)

# fc layer, weight needs to be tansposed
fc_names = ["layers.0.0.fn.to_qkv.weight","layers.0.0.fn.to_q_attn_logits.weight",
            "layers.0.0.fn.to_k_attn_logits.weight", "layers.0.0.fn.to_r.weight",
            "layers.0.0.fn.to_out.weight", "layers.0.1.fn.0.weight",
            "layers.0.1.fn.2.weight","layers.1.0.fn.to_qkv.weight",
            "layers.1.0.fn.to_q_attn_logits.weight","layers.1.0.fn.to_k_attn_logits.weight",
            "layers.1.0.fn.to_r.weight","layers.1.0.fn.to_out.weight",
            "layers.1.1.fn.0.weight","layers.1.1.fn.2.weight",
            "to_logits.1.weight"]
print("Total FC Layers: {}".format(len(fc_names)))

count_fc = 0
for key in torch_checkpoint:
    params = torch_checkpoint[key].cpu().detach().numpy()
    flag = [item in key for item in fc_names]
    # FC Trans
    if any(flag):
        params = params.transpose()
        count_fc = count_fc + 1
        paddle_dict[key] = params
        continue
    # normal
    paddle_dict[key] = params
print("{} FC Layer Params Transposed".format(count_fc))
paddle.save(paddle_dict,paddle_chkpt_path)
paddle_checkpoint = paddle.load(paddle_chkpt_path)
print(1)

