import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil

#! see here: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py
# Params
# RunningAverage
# load/save state_dict
# metrics

class Params():
    """
        load and save json consisting of parameter settings with Params
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            #use dict A to update dict B
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)


    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            #use dict A to update dict B
            self.__dict__.update(params)

class RunningAverage():

    def __init__(self):
        self.total = 0.0
        self.step = 0

    def update(self, val):
        self.total += val
        self.step += 1
    
    def __call__(self):
        return self.total / float(self.step)

def save_checkpoint(state, is_best, checkpoint):
    #* Here checkpoint is the folder including parameter (last and best)
    file_path = os.path.join(checkpoint, "last.pth.tar")
    if (not os.path.exists(checkpoint)):
        print("Dictionary does not exist! Creating dictionary {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Dictionary exists!")

    # save dict() into .pth.tar (only for torch)
    torch.save(state, file_path)

    if (is_best):
        shutil.copyfile(file_path, os.path.join(checkpoint, "best.pth.tar"))


def load_checkpoint(checkpoint, model, optimizer=None, cuda_id=0):
    """ Load parameters from file into model and optimizer
    
    Args:
        *checkpoint: file to be loaded (slightly different with the meaning in save_checkpoint)
        model: model to be loaded
        optimizer: optimizer to be loaded
    """

    if (not os.path.exists(checkpoint)):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location="cuda:"+str(cuda_id))
    model.load_state_dict(checkpoint['state_dict'])
    print("parameter is loaded")
    if (optimizer is not None):
        optimizer.load_state_dict(checkpoint['optim_dict'])
    
    return model, checkpoint
    #* more convenient if we want to check info in checkpoint

def accuracy(batch_pred, batch_label, params):
    batch_pred = torch.sigmoid(batch_pred)
    threshold = params.threshold
    return (((batch_pred > threshold) == batch_label).sum().item()) \
        / (params.batch_size * params.output_size)

metrics = {
    "accuracy":accuracy
}
