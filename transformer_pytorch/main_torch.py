import torch
from transformer_pytorch import FastTransformer
import numpy as np
from reprod_log import ReprodLogger


reprod_logger = ReprodLogger()
seed = 42
torch.manual_seed(seed)

model = FastTransformer(
    num_tokens = 20000,
    dim = 512,
    depth = 2,
    max_seq_len = 4096,
    absolute_pos_emb = True   # default uses relative positional encoding, but if that isn't working, then turn on absolute positional embedding by setting this to True
)

model_dict = model.state_dict()
# for key, param in model_dict.items():
#     print(key)
torch.save(model_dict,"torch_init.pth")

x_np = np.load('../fake_data/x.npy')
x = torch.from_numpy(x_np)
mask_np= np.load('../fake_data/mask.npy')
mask = torch.from_numpy(mask_np).bool()

logits = model(x, mask = mask) # (1, 4096, 20000)

reprod_logger.add("logits", logits.cpu().detach().numpy())
reprod_logger.save("forward_torch.npy")
