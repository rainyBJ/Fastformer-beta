import os
print(os.getcwd())
os.chdir("/root/baidu/paddle/fast_transformer_pytorch/transformer_paddle")

import paddle
from fast_transformer_pd import FastTransformer
import numpy as np

seed = 42
paddle.seed(seed)

model = FastTransformer(num_tokens=20000, dim=512, depth=2, max_seq_len=\
    4096, absolute_pos_emb=True)
# 模型初始化对齐
model_dict = model.state_dict()
torch_model_dict = paddle.load("paddle_init.pdparams")
torch_model_dict = {k: v for k, v in torch_model_dict.items() if k in model_dict}
model_dict.update(torch_model_dict)
for key, param in model_dict.items():
    print(key)

model.load_dict(model_dict)
# 输入数据对齐
x_np = np.load('../fake_data/x.npy')
x = paddle.to_tensor(x_np)
mask_np= np.load('../fake_data/mask.npy')
mask = paddle.to_tensor(mask_np)
logits = model(x, mask=mask)
print(logits)
