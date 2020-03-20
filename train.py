import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from dataset import MixedSignalDataset
from utils import TrainerLogger
from tqdm import trange

from model import MultiCLDNN

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

EPISODE = 10
LR = 1e-5

model = MultiCLDNN(kernel_size=8).to(device)
optimizer = optim.Adam(model.parameters(), lr = LR)
dataset = MixedSignalDataset(device=device)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
loss_fn = F.binary_cross_entropy_with_logits
trainer_logger = TrainerLogger()

print("data loaded!")
print("-"*30)

for episode in trange(EPISODE):
    for sample in dataloader:
        feature = sample[0]
        logit = sample[1]
        pred = model(feature)
        loss = loss_fn(pred, logit)
        loss.backward()
        optimizer.step()
    trainer_logger.log(loss)
    print(f"training loss:\t{loss}")
trainer_logger.plot()