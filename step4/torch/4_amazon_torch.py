from model_torch import Model
import torch
import numpy as np
import torch.optim as optim
from criterion_torch import acc
from reprod_log import ReprodLogger
reprod_logger = ReprodLogger()

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# load data
data = np.load("../../dataset/data.npy")
label = np.load("../../dataset/label.npy")
num_tokens = int(np.load("../../dataset/num_tokens.npy"))

# model_initial and export params
model = Model()
state_dict = model.state_dict()
# torch.save(state_dict,'initial_torch.pth')

# 优化器
optimizer = optim.Adam([{'params': model.parameters(), 'lr': 1e-3}])

model = model.cuda()

# split dataset
total_num = int(len(label))
train_num = int(total_num * 0.8)
val_num = int(total_num * 0.9)
index = np.arange(total_num)
train_index = index[:train_num]
val_index = index[train_num:val_num]
test_index = index[val_num:]

epochs = 1
loss_list = []

for epoch in range(epochs):
    loss = 0.0
    macrof = 0.0
    accuary = 0.0
    np.random.shuffle(train_index) # 每个 epoch shuffle
    for cnt in range(len(train_index) // 64):

        # 保存前五个 step 的 loss 用于反向对齐
        if cnt == 5:
            break

        log_ids = data[train_index][cnt * 64:cnt * 64 + 64, :512]
        targets = label[train_index][cnt * 64:cnt * 64 + 64]

        log_ids = torch.LongTensor(log_ids).cuda(non_blocking=True)
        targets = torch.LongTensor(targets).cuda(non_blocking=True)
        bz_loss, y_hat = model(log_ids, targets)
        loss_list.append(bz_loss.cpu().detach())
        accuary += acc(targets, y_hat)[0]
        unified_loss = bz_loss
        optimizer.zero_grad()
        unified_loss.backward()
        optimizer.step()

reprod_logger.add("loss", np.array(loss_list))
reprod_logger.save("4_step_loss_torch.npy")