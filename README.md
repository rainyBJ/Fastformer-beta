# Fast Transformer - Paddle

## TD

[paddle 复现论文参考 repo](https://github.com/PaddlePaddle/models/blob/develop/docs/ThesisReproduction_CV.md)

- [ ] step1 模型结构对齐
  - [x] 模型输入对齐，生成 fake_data
  - [x] 模型初始化对齐
  - [ ] 模型前向传播对齐

## 文件结构

```
fast-transformer-pytorch-main
├─ LICENSE
├─ README.md
├─ pics
│    └─ debug.py
├─ chkpt_initial
│    └─ chkpt_convert.py
├─ fake_data
│    └─ gen_fake_data.py
├─ fast-transformer.png
├─ setup.py
├─ step1
├─ test.py
├─ transformer_paddle
│    ├─ __init__.py
│    ├─ fast_transformer_pd.py
│    └─ main_pd.py
└─ transformer_pytorch
      ├─ __init__.py
      ├─ fast_transformer_torch.py
      └─ main_torch.py
```

- pics 存放图片
- chkpt_initial 模型初始化对齐
- fake_data 模型输入对齐
- setup.py 依赖环境
- **step1** 模型结构对齐，待完善
- **transformer_paddle** 
  - fast_transformer_pd.py 待转换 paddle 代码
  - main_pd.py 顶层测试用，其中路径需改动
- transformer_pytorch 
  - fast_transformer_torch.py 原 torch 代码
  - main_torch.py 顶层测试用，其中路径需改动

## debug

debug 视图

- 设置断点，看对应变量、运算过程是否相同

![debug](pics/debug.png)

# 原 repo

<img src="./fast-transformer.png" width="400px"></img>

## Fast Transformer - Pytorch

Implementation of <a href="https://arxiv.org/abs/2108.09084">Fast Transformer</a> in Pytorch. This only work as an encoder.

<a href="https://www.youtube.com/watch?v=qgUegkefocg">Yannic video</a>

<a href="https://www.youtube.com/watch?v=Ich5TIvdYRE">AI Epiphany</a>

## Install

```bash
$ pip install fast-transformer-pytorch
```

## Usage

```python
import torch
from transformer_pytorch import FastTransformer

model = FastTransformer(
    num_tokens=20000,
    dim=512,
    depth=2,
    max_seq_len=4096,
    absolute_pos_emb=True
    # default uses relative positional encoding, but if that isn't working, then turn on absolute positional embedding by setting this to True
)

x = torch.randint(0, 20000, (1, 4096))
mask = torch.ones(1, 4096).bool()

logits = model(x, mask=mask)  # (1, 4096, 20000)
```

## Citations

```bibtex
@misc{wu2021fastformer,
    title   = {Fastformer: Additive Attention is All You Need}, 
    author  = {Chuhan Wu and Fangzhao Wu and Tao Qi and Yongfeng Huang},
    year    = {2021},
    eprint  = {2108.09084},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```
