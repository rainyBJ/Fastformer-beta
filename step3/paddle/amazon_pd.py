import paddle
import numpy as np
import paddle.optimizer as optim
from model_pd import Model
from criterion_pd import acc

from reprod_log import ReprodLogger
reprod_logger = ReprodLogger()

# 设置随机种子
seed = 42
paddle.seed(seed)
np.random.seed(seed)

# load data
data = np.load("../../dataset/data.npy")
label = np.load("../../dataset/label.npy")
num_tokens = int(np.load("../../dataset/num_tokens.npy"))

# longTensor
def LongTensor(x):
    x = paddle.to_tensor(x, dtype="int64")
    return x

# load model and state_dict
model = Model()
model_dict = model.state_dict()
torch_model_dict = paddle.load("best_val.pdparams")
torch_model_dict = {k: v for k, v in torch_model_dict.items() if k in model_dict}
model_dict.update(torch_model_dict)
model.load_dict(model_dict)

# 优化器
optimizer = optim.Adam(parameters=model.parameters(),learning_rate=1e-3)

# split dataset
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
        log_ids = LongTensor(log_ids)
        targets = LongTensor(targets)

        bz_loss2, y_hat2 = model(log_ids, targets)
        allpred += y_hat2.detach().numpy().tolist()

    y_pred = np.argmax(allpred, axis=-1)
    y_true = label[val_index]
    metric = acc(paddle.to_tensor(y_true), paddle.to_tensor(y_pred),eval=True)
    acc_val = round(float(metric[0]), 4)
    print("accuracy: ")
    print(acc_val)
    macrof_val = round(metric[1], 4)
    print("macrof: ")
    print(macrof_val)

    reprod_logger.add("macrof", np.array(macrof_val))
    reprod_logger.save("best_val_pd.npy")