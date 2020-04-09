import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.dataset import MixedSignalIQDataset
from torch.utils.data.dataloader import DataLoader
from model.lstm_net import LSTMNet
from tqdm import tqdm
from tensorboardX import SummaryWriter

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

num_epoch = 10000
lr = 1e-3 //1e-3
batch_size = 1024

hidden_size = 128
num_layers = 2
num_class = 11
threshold = 0.5
data_dir = "./data/train_data_400_with_high_snr.h5"

model = LSTMNet(hidden_size=hidden_size, num_layers=num_layers,
                batch_size=batch_size, output_size=num_class).to(device)
optimizer = optim.RMSprop(model.parameters(), lr = lr)
dataset = MixedSignalIQDataset(data_dir=data_dir, device=device)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("loaded into DataLoader!")
#! different usage of entropy related loss: https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html
loss_fn = F.binary_cross_entropy_with_logits

for epoch in range(num_epoch):
    training_loss = 0.0
    training_sample = 0
    correct = 0

    tloader = tqdm(dataloader)

    for (X_batch, y_batch) in tloader:
        if (y_batch.shape[0] != batch_size): continue
        logit = model(X_batch)
        loss = loss_fn(logit, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        correct += ((torch.sigmoid(logit) > threshold)== y_batch).sum().item()
        training_sample += batch_size

    training_loss /= training_sample
    accuracy = correct / (num_class * training_sample)
    #! output format: https://www.runoob.com/python/att-string-format.html
    print("[epoch {:>4d}] train_loss:{:>8.5f} accuracy:{:8.5f}".format(epoch, training_loss, accuracy))