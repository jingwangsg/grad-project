from torch.utils.data.dataset import Dataset
import h5py

import _pickle as pickle

class MixedSignalDataset(Dataset):

    def __init__(self):
        super(MixedSignalDataset).__init__()
        with h5py.File("./data/processed_data.h5", "r") as f:
            self.feature_mat = f.get("feature_mat")
            self.label_mat = f.get("logit_mat")
        self.len = self.feature_mat.shape[0]
    
    def __getitem__(self, idx):
        return (self.feature_mat[idx], self.logit_mat[idx])
    
    def __len__(self):
        return self.len