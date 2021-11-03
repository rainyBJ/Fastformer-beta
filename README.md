# Fast Transformer - Paddle

## TD

- [x] step1 模型结构对齐
  - [x] 模型输入对齐，生成 fake_data
  - [x] 模型初始化对齐
  - [x] 模型前向传播对齐
  
- [x] step3 模型在训练数据上 loss、criterion 对齐
  - [x] 输入数据对齐
  - [x] 参数对齐
  - [x] 指标对齐
  
- [ ] Step 4 反向对齐

  - 基本上能对齐，最后一个 linear 层 bias 更新有问题

  - 前五个 step，loss 的 averge diff 为 10e-3 量级

- [x] step5 模型训练
  - [x] 输入数据集对齐
  - [x] 模型初始化对齐 
  - [x] MacoF达到论文指标
  - [x] 数据集类别平衡
  - [x] 加入 dropout=0.2

## 文件结构

```
fast-transformer-pytorch-main
├─ LICENSE
├─ README.md
├─ chkpt_initial
│    └─ chkpt_convert.py
├─ dataset
│    └─ datasetloader.py
├─ fake_data
│    ├─ gen_fake_data.py
│    ├─ mask.npy
│    └─ x.npy
├─ fast-transformer.png
├─ pics
│    ├─ debug.png
│    ├─ image-20211024144240527.png
│    ├─ image-20211024144336758.png
│    ├─ 反向对齐1.png
│    ├─ 反向对齐2.png
│    └─ 超惨设置.png
├─ setup.py
├─ step1
│    ├─ check_step1.py
│    ├─ chkpt_convert_initial.py
│    ├─ forward_diff.log
│    ├─ paddle
│    │    ├─ fast_transformer_pd.py
│    │    └─ pd_forward.py
│    └─ torch
│           ├─ fast_transformer_torch.py
│           └─ torch_forward.py
├─ step3
│    ├─ best_val.log
│    ├─ check_step3.py
│    ├─ chkpt_convert_best_val.py
│    ├─ paddle
│    │    ├─ amazon_pd.py
│    │    ├─ criterion_pd.py
│    │    ├─ fast_transformer_pd.py
│    │    └─ model_pd.py
│    └─ torch
│           ├─ amazon_torch.py
│           ├─ criterion_torch.py
│           ├─ fast_transformer_torch.py
│           └─ model_torch.py
├─ step4
│    ├─ check_step4.py
│    ├─ loss_check.log
│    ├─ paddle
│    │    ├─ 4_amazon_pd.py
│    │    ├─ 4_step_loss_pd.npy
│    │    ├─ criterion_pd.py
│    │    ├─ fast_transformer_pd.py
│    │    └─ model_pd.py
│    └─ torch
│           ├─ 4_amazon_torch.py
│           ├─ 4_step_loss_torch.npy
│           ├─ criterion_torch.py
│           ├─ fast_transformer_torch.py
│           └─ model_torch.py
├─ step5
│    ├─ paddle
│    │    ├─ 5_amazon_pd.py
│    │    ├─ criterion_pd.py
│    │    ├─ fast_transformer_pd.py
│    │    ├─ log_pd.txt
│    │    └─ model_pd.py
│    └─ torch
│           ├─ 5_amazon_torch.py
│           ├─ criterion_torch.py
│           ├─ fast_transformer_torch.py
│           ├─ log_torch.txt
│           └─ model_torch.py
└─ test.py
```

- pics 存放图片
- chkpt_initial 参数转换
- dataset 随机生成 50k 训练集，5k 测试集数据
- fake_data 产生模型输入
- setup.py 依赖环境
- **step1** 模型结构对齐
- **step3** 验证集上指标对齐
- **step4** 反向对齐
- **step5** 训练对齐

## 问题

### 网络最后一层梯度对齐

**怀疑是 paddle 的 bug，实际上对齐了但是显示错误**

- bias 不可
- weight 可以

### 环境

- 模型参数对齐、输入数据对齐、dropout为0、参数初始化（直接导入了PyTorch中的初始化参数）

### 差异

Torch

![img](pics/反向对齐2.png)

Paddle

![img](pics/反向对齐1.png)

## 任务

Amazon Electronic Review Rating Classification

文本分类任务， 5 分类，随机抽取 Electronics 子类中 50k 的数据集

- 40k train
- 5k eval
- 5k test

### 情感主题分类

- F1

![image-20211024144240527](pics/image-20211024144240527.png)



## 论文

### 网络结构

<img src="./fast-transformer.png" width="400px"></img>

### 实验设置

![image-20211024144120831](pics/超惨设置.png)

## AI Studio 项目链接

https://aistudio.baidu.com/aistudio/projectdetail/2559430?shared=1

## 参考 repo

- https://github.com/wilile26811249/Fastformer-PyTorch

- https://github.com/wuch15/Fastformer
