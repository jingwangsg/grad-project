import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.dataset import MixedSignalDataset
from torch.utils.data.dataloader import DataLoader
from model.gru_net import GRUNet
from model.lstm_net import LSTMNet
from model.cldnn import CLDNN
import model.dataset as dataset
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import RunningAverage, load_checkpoint, save_checkpoint, metrics
import utils
import os
import numpy as np
from tensorboardX import SummaryWriter

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#! https://www.cnblogs.com/ying-chease/p/9473938.html
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

#! How to use TensorBoardX https://www.jianshu.com/p/46eb3004beca
writer = None
args = None

# model = LSTMNet(hidden_size=hidden_size, num_layers=num_layers,
#                 batch_size=batch_size, output_size=num_class).to(device)
# optimizer = optim.Adam(model.parameters(), lr = lr)
# dataset = MixedSignalIQDataset(data_dir=data_dir, device=device)
# num_training_sample = int(dataset.__len__ * train_test_ratio )
# num_validation_sample = dataset.__len__ - num_training_sample
# #! random_split to directly split "dataset": https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
# training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
# validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
# print("loaded into DataLoader!")
# #! different usages of entropy related loss: https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html
# loss_fn = F.binary_cross_entropy_with_logits


def train(model, dataloader, optimizer, loss_fn, metrics, params, epoch):
    model.train()

    tloader = tqdm(dataloader)
    loss_avg = RunningAverage()
    batch_sums = []
    #! https://discuss.pytorch.org/t/cyclic-learning-rate-how-to-use/53796

    for (X_batch, y_batch) in tloader:
        if (X_batch.shape[0] != params.batch_size): continue
        logit = model(X_batch)
        loss = loss_fn(logit, y_batch)
        #! model.zero_grad() v.s. optimizer.zero_grad() https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/4
        # when all parameters are in optimizer, model.zero_grad() is the same as optimizer.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        loss_avg.update(loss.item())
        cur_batch_sum = {metric: metrics[metric](logit, y_batch, params) for metric in metrics}
        cur_batch_sum["loss"] = loss.item()
        batch_sums.append(cur_batch_sum)
        tloader.set_postfix(loss='{:05.3f}'.format(loss_avg()) )
    
    metric_with_loss = list(metrics.keys()) + ["loss"]
    metric_means = {metric: np.mean([x[metric] for x in batch_sums]) for metric in metric_with_loss}
    metric_str = " ; ".join("{}: {:05.3f}".format(k, v)\
                                for (k, v) in metric_means.items())
    print("- Train metrics: " + metric_str)
    for (k, v) in metric_means.items():
        writer.add_scalar("scalar/train/"+k, v, epoch)

def evaluate(model, dataloader, loss_fn, metrics, params, epoch):
    model.eval()

    tloader = tqdm(dataloader)
    batch_sums = []

    for (X_batch, y_batch) in tloader:
        if (X_batch.shape[0] != params.batch_size): continue
        logit = model(X_batch)
        loss = loss_fn(logit, y_batch)

        cur_batch_sum = {metric: metrics[metric](logit, y_batch, params) for metric in metrics}
        cur_batch_sum["loss"] = loss.item()
        batch_sums.append(cur_batch_sum)

    metric_with_loss = list(metrics.keys()) + ["loss"]
    metric_means = {metric: np.mean([x[metric] for x in batch_sums]) for metric in metric_with_loss}
    metric_str = " ; ".join("{}: {:05.3f}".format(k, v)\
                                for (k, v) in metric_means.items())
    print("- Test metrics: " + metric_str)
    for (k, v) in metric_means.items():
        writer.add_scalar("/scalar/test/" + k, v, epoch)

    #* return metric to decide whether it's the best result during epochs
    return metric_means

def train_and_evaluate(model, train_dataset, val_dataset, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    #* The params is explicitly needed because we may add some parameters about dataset like train_size into it after reading it
    """Base function to train models

    Args:
        model: (torch.nn.Module) module to be trained
        train_data, val_data: (torch.utils.data.Dataset)
        optimizor: (torch.optim) 
        loss_fn: loss function for back propogation
        params: (Params)
        model_dir: (String) dir path of whole model
        restore_file: (torch.state_dict) dict including epoch, state_dict, optimizer state
    """

    if (restore_file is not None):
        print("Restoring parameters from {}".format(restore_file))
        load_checkpoint(restore_file, model, optimizer)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        #! https://www.runoob.com/python/att-string-format.html
        print("[Epoch {}/{}]".format(epoch, params.num_epochs))
        train(model, train_dataloader, optimizer, loss_fn, metrics, params, epoch)
        val_metric = evaluate(model, val_dataloader, loss_fn, metrics, params, epoch)

        val_acc = val_metric["accuracy"]

        is_best = val_acc > best_val_acc
        if (is_best):
            best_val_acc = val_acc

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        }, is_best=is_best, checkpoint=model_dir)

        if (is_best):
            # print all val_metric into json
            pass

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument("--data_dir")
parser.add_argument("--model_dir")
parser.add_argument("--restore_file", default=None)
parser.add_argument("--gpu_id", default="0");

if (__name__ == "__main__"):
    """read from argument and call functions above to train and evaluate certain model setting in corresponding experiment folder
    
    Args from arg_parser:
        --data_dir: using which dataset
        --model_dir: training and evaluating which model
        --restore_file: load weight according to which parameter file (if any)
    """
    args = parser.parse_args()
    device = torch.device("cuda:"+args.gpu_id)
    print(device)
    log_path = os.path.join("./logs", args.model_dir)
    writer = SummaryWriter(log_dir="./logs/"+args.model_name)
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.exists(json_path), "No json file found at {}".format(json_path)
    params = utils.Params(json_path)

    raw_data = dataset.load_data(args.data_dir, device)
    print("Data loaded in memory")
    # split data into train dataset and test dataset
    raw_dataset = MixedSignalDataset(raw_data, device)
    num_sample = raw_dataset.len
    
    num_train_sample = int(params.train_test_ratio * num_sample)
    splited_len = [num_train_sample, num_sample - num_train_sample]
    train_dataset, val_dataset = torch.utils.data.random_split(raw_dataset, splited_len)

    models = {"LSTMNet": LSTMNet, "GRUNet": GRUNet, "CLDNN": CLDNN}
    model_type = args.model_name.split("_")[0]
    model_class = models[model_type]
    model = model_class(params, device).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = F.binary_cross_entropy_with_logits

    # Train the model
    print("Starting training for {} epochs".format(params.num_epochs) )
    train_and_evaluate(model, train_dataset, val_dataset, optimizer, loss_fn, metrics, params, args.model_dir, args.restore_file)
