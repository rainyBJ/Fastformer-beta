from model_torch import Model
import torch
import numpy as np
import torch.optim as optim
from criterion_torch import acc

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

epochs = 3
with open('log_torch.txt','w') as f:
    for epoch in range(epochs):
        loss = 0.0
        macrof = 0.0
        accuary = 0.0
        np.random.shuffle(train_index) # 每个 epoch shuffle
        for cnt in range(len(train_index) // 64):

            log_ids = data[train_index][cnt * 64:cnt * 64 + 64, :512]
            targets = label[train_index][cnt * 64:cnt * 64 + 64]

            log_ids = torch.LongTensor(log_ids).cuda(non_blocking=True)
            targets = torch.LongTensor(targets).cuda(non_blocking=True)
            bz_loss, y_hat = model(log_ids, targets)
            loss += bz_loss.data.float()
            accuary += acc(targets, y_hat)[0]
            unified_loss = bz_loss
            optimizer.zero_grad()
            unified_loss.backward()
            optimizer.step()

            # 打印梯度
            # for name, tensor in model.named_parameters():
            # for name, tensor in model.named_parameters():
            #     grad = tensor.grad
            #     print(name)
            #     try:
            #         print(grad.shape)
            #         print(grad)
            #         print(10*"*")
            #     except:
            #         print(10 * "*")
            if cnt % 10 == 0:
                print(
                    ' Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(cnt * 64, loss.data / (cnt + 1), accuary / (cnt + 1))
                )
                f.write(
                    ' Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(cnt * 64, loss.data / (cnt + 1),
                                                                      accuary / (cnt + 1)) + '\n'
                )
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
        f.write("accuracy: \n")
        print(acc_val)
        f.write(str(acc_val)+'\n')
        macrof_val = round(metric[1], 4)
        print("macrof: ")
        f.write("macrof: \n")
        print(macrof_val)
        f.write(str(macrof_val)+"\n")
        # save params
        if macrof_val > macrof:
            state_dict = {}
            state_dict['macrof'] = macrof_val
            state_dict['model'] = model.state_dict()
            torch.save(state_dict,"best_val.pth")
        if epoch == epochs - 1:
            state_dict = {}
            state_dict['macrof'] = macrof_val
            state_dict['model'] = model.state_dict()
            torch.save(state_dict, "last_epoch.pth")
        model.train()