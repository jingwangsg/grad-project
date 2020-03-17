import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from dataset import MixedSignalDataset

from model import MultiCLDNN

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

EPISODE = 1000
LR = 1e-5

model = MultiCLDNN(kernel_size=8).to(device)
optimizer = optim.Adam(model.parameters(), lr = LR)
dataset = MixedSignalDataset()
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
loss_fn = 

for episode in range(EPISODE):
    for sample in dataloader:
        feature = sample[0]
        label = sample[1]
        pred = model(feature)
