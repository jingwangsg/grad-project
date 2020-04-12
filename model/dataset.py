from torch.utils.data.dataset import Dataset
import torch
import h5py

class MixedSignalDataset(Dataset):

    def __init__(self, data_dict, device):
        super(MixedSignalDataset).__init__()

        self.feature_mat = data_dict["data"]
        self.logit_mat = data_dict["label"]

        self.len = self.feature_mat.shape[0]

    def __getitem__(self, idx):
        return (self.feature_mat[idx], self.logit_mat[idx])
    
    def __len__(self):
        return self.len

def load_data(data_dir, device):
    data_dict = dict()
    with h5py.File(data_dir, "r") as f:
        data_dict["data"] = torch.Tensor(f["feature_mat"]).to(device)
        data_dict["data"] = data_dict["data"].permute([0, 2, 1])
        data_dict["label"] = torch.Tensor(f["logit_mat"]).to(device)
        print("loading\t{}...".format(data_dir))
        print("load feature_mat:\t{}".format(data_dict["data"].shape))
        print("load logit_mat:\t\t{}".format(data_dict["label"].shape))
    return data_dict
    
