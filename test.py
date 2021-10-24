import torch
import numpy as np
import paddle

# tensor.tile tensor.where
data_np = np.random.rand(1,2,2).astype("float32")
mask_np = np.ones(shape=[1,1,2],dtype='bool')
mask_np[0][0][0] = 0
mask_value = 100
data_torch = torch.from_numpy(data_np)
print(data_torch)
mask_torch = torch.from_numpy(mask_np)
print(mask_torch)
output_torch = data_torch.masked_fill(mask_torch, mask_value)
print(output_torch)

print(10*"*")

data_pd = paddle.to_tensor(data_np)
print(data_pd)
mask_pd = paddle.to_tensor(mask_np)
print(mask_pd)
mask_pd = mask_pd.tile([1,2,1])
print(mask_pd)

print(1)
