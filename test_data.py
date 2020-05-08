import _pickle as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.lstm_net import LSTMNet
from model.dataset import MixedSignalDataset
from utils import Params, save_checkpoint, load_checkpoint, metrics

with open("./data/test_data_5_100.h5", "rb") as f:
    data = pickle.load(f, encoding="iso-8859-1")

feature_mat = data[0]
logit_mat = data[1]

exp_dir = "./experiment/LSTMNet/wider_or_deeper/256x2/"

params = Params(exp_dir + "params.json")
device = torch.device("cuda")

model = LSTMNet(params, device).to(device)
model.eval()
model, _ = load_checkpoint(exp_dir + "best.pth.tar", model)
loss_fn = F.binary_cross_entropy_with_logits
accuracy_fn = metrics["accuracy"]


for snr in feature_mat.keys():
    data_dict = {}
    data_dict["data"] = torch.from_numpy(feature_mat[snr]).to(device)
    data_dict["label"] = torch.from_numpy(logit_mat[snr]).to(device)
    dataset = MixedSignalDataset(data_dict, device)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
    tloader = tqdm(dataloader)

    num_batch = 0
    avg_acc = 0.0

    for (X_batch, y_batch) in tloader:
        if (X_batch.shape[0] != params.batch_size): continue
        logit = model(X_batch)
        acc = accuracy_fn(logit, y_batch, params)
        avg_acc += acc
        num_batch += 1

    avg_acc = avg_acc / num_batch

    print("{:d}:{:05.3f}".format(snr, avg_acc))

