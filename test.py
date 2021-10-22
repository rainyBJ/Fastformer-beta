import torch
import numpy as np
import paddle

data_np = np.random.rand(3,4).astype("float32")
data_torch = torch.from_numpy(data_np)
data_paddle = paddle.to_tensor(data_np)

print(1)
