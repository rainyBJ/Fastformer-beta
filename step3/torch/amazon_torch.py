import torch
import numpy as np
import torch.optim as optim
from  model_torch import Model
from  criterion_torch import acc

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

# # load model and state_dict
model = Model()
state_dict = torch.load("best_val.pth")['model']
model.load_state_dict(state_dict)
optimizer = optim.Adam([{'params': model.parameters(), 'lr': 1e-3}])
model = model.cuda()

# train val test dataset
total_num = int(len(label))
train_num = int(total_num * 0.8)
val_num = int(total_num * 0.9)
index = np.arange(total_num)
train_index = index[:train_num]
val_index = index[train_num:val_num]
test_index = index[val_num:]

# evaluate for 1 epoch
epochs = 1
for epoch in range(epochs):
    model.eval()
    allpred = []
    for cnt in range(len(val_index) // 64 + 1):
        log_ids = data[val_index][cnt * 64:cnt * 64 + 64, :256]
        targets = label[val_index][cnt * 64:cnt * 64 + 64]
        log_ids = torch.LongTensor(log_ids).cuda(non_blocking=True)
        targets = torch.LongTensor(targets).cuda(non_blocking=True)

        bz_loss2, y_hat2 = model(log_ids, targets)
        allpred += y_hat2.to('cpu').detach().numpy().tolist()

    y_pred = np.argmax(allpred, axis=-1)
    y_true = label[val_index]
    metric = acc(torch.from_numpy(y_true), torch.from_numpy(y_pred),eval=True)
    acc_val = round(float(metric[0]), 4)
    print("accuracy: ")
    print(acc_val)
    macrof_val = round(metric[1], 4)
    print("macrof: ")
    print(macrof_val)

    reprod_logger.add("macrof", np.array(macrof_val))
    reprod_logger.save("best_val_torch.npy")