from torch.utils.data.dataset import Dataset
import torch
import h5py

import _pickle as pickle

class MixedSignalDataset(Dataset):

    def __init__(self, device):
        super(MixedSignalDataset).__init__()
        with h5py.File("./data/processed_data.h5", "r") as f:
            self.feature_mat = torch.Tensor(f["feature_mat"])
            self.feature_mat = torch.unsqueeze(self.feature_mat, dim=1)
            self.feature_mat = self.feature_mat.to(device)
            self.logit_mat = torch.Tensor(f["logit_mat"])
            self.logit_mat = self.logit_mat.to(device)
        print(f"load feature_mat:\t{self.feature_mat.shape}")
        print(f"load logit_mat:\t\t{self.logit_mat.shape}")
        self.len = self.feature_mat.shape[0]
    
    def __getitem__(self, idx):
        return (self.feature_mat[idx], self.logit_mat[idx])
    
    def __len__(self):
        return self.len