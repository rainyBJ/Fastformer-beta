import random

import torch
import numpy as np
import paddle

'''
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
'''

'''
# random seed
import random
seed = 42
random.seed(seed)
x = range(1689188)
y = random.sample(x,50000)
print(x[0])
'''

'''
保留两位小数
a = 0.12345678
b = round(a,2)
print("num:{:.4f}".format(a))
'''

data_np = np.random.rand(2,5).astype("float32")
data_paddle_1 = paddle.to_tensor([[1,2,3,4],
                                [5,6,7,8]])
data_paddle_2 = paddle.to_tensor([[1,6,7,4],
                                 [5,6,7,8]])
sum = paddle.sum(data_paddle_2==data_paddle_1, axis=-1)
print(1)