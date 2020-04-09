from torch.utils.data.dataset import Dataset
import torch
import h5py

import _pickle as pickle

class MixedSignalIQDataset(Dataset):

    def __init__(self, data_dir, device):
        super(MixedSignalIQDataset).__init__()
        with h5py.File(data_dir, "r") as f:
            self.feature_mat = torch.Tensor(f["feature_mat"])
            # self.feature_mat = torch.unsqueeze(self.feature_mat, dim=1)
            # in LSTM we need input with (batch, seq_len, input_size)
            self.feature_mat = self.feature_mat.to(device)
            # we need to modify axis of feature_mat (batch, input_size, seq_len) into (batch, seq_len, input_size)
            #! permute axis: https://www.cnblogs.com/yifdu25/p/9399047.html
            self.feature_mat = self.feature_mat.permute(0, 2, 1)
            self.logit_mat = torch.Tensor(f["logit_mat"])
            self.logit_mat = self.logit_mat.to(device)
        print(f"load feature_mat:\t{self.feature_mat.shape}")
        print(f"load logit_mat:\t\t{self.logit_mat.shape}")
        self.len = self.feature_mat.shape[0]
    
    def __getitem__(self, idx):
        return (self.feature_mat[idx], self.logit_mat[idx])
    
    def __len__(self):
        return self.len
