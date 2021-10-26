from transformer_pytorch.fast_transformer_torch import FastTransformer
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import  os
print(os.getcwd())
# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# load data
data = np.load("dataset/data.npy")
label = np.load("dataset/label.npy")
num_tokens = int(np.load("dataset/num_tokens.npy"))


# accuracy & macro-f
def acc(y_true, y_hat, eval=False):
    if eval:
        y_hat = y_hat
    else:
        y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    # accuracy
    hit = torch.sum(y_true == y_hat)
    f = 0
    if eval:
        # F-macro
        eps = 1e-8
        TP = {}
        FN = {}
        FP = {}
        TN = {}
        precision_dict = {}
        recall_dict = {}
        F = {}
        for cls in range(5):
            TP[cls] = 0
            FN[cls] = 0
            FP[cls] = 0
            TN[cls] = 0
            precision_dict[cls] = 0
            recall_dict[cls] = 0
            F[cls] = 0
        for i in range(len(y_true)):
            predict = y_hat[i]
            true = y_true[i]
            for cls in range(5):
                if true==cls and predict==cls:
                    TP[cls] = TP[cls] + 1
                elif true==cls and predict!=cls:
                    FN[cls] = FN[cls] + 1
                elif true!=cls and predict==cls:
                    FP[cls] = FP[cls] + 1
                else:
                    TN[cls] = TN[cls] + 1

        for cls in range(5):
            precision_dict[cls] = TP[cls] / (TP[cls] + FP[cls] + eps)
            recall_dict[cls] = TP[cls] / (TP[cls] + FN[cls] + eps)
        precision = 0
        recall = 0
        for cls in range(5):
            precision = precision + precision_dict[cls]
            recall = recall + recall_dict[cls]
        precision = precision/5
        recall = recall/5
        f = 2*precision*recall/(precision+recall)

    return  hit.data.float() * 1.0 / tot, f

# model
model = FastTransformer(
    num_tokens = num_tokens,
    dim = 512,
    depth = 2,
    max_seq_len = 512,
    absolute_pos_emb = True,
    dropout=0 # default uses relative positional encoding, but if that isn't working, then turn on absolute positional embedding by setting this to True
)

class Model(torch.nn.Module):

    def __init__(self, ):
        super(Model, self).__init__()
        self.dense_linear = nn.Linear(512, 5)
        self.fastformer_model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, targets):
        mask = input_ids.bool()
        text_vec = self.fastformer_model(input_ids, mask)
        score = self.dense_linear(text_vec)
        loss = self.criterion(score, targets)
        return loss, score

# train
model = Model()
state_dict = model.state_dict()
for key, param in state_dict.items():
    print(key)
# state_dict = model.state_dict()
# torch.save(state_dict,'initial_torch.pth')
optimizer = optim.Adam([{'params': model.parameters(), 'lr': 1e-3}])
model = model.cuda()

total_num = int(len(label))
train_num = int(total_num * 0.8)
val_num = int(total_num * 0.9)
index = np.arange(total_num)
train_index = index[:train_num]
val_index = index[train_num:val_num]
test_index = index[val_num:]

epochs = 3
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
        for name, tensor in model.named_parameters():
            grad = tensor.grad
            print(name)
            try:
                print(grad.shape)
                print(grad)
                print(10 * "*")
            except:
                print(10 * "*")
        optimizer.step()
        if cnt % 10 == 0:
            print(
                ' Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(cnt * 64, loss.data / (cnt + 1), accuary / (cnt + 1)))
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