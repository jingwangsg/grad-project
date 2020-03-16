from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import _pickle as pickle

class MixedSignalDataset(Dataset):

    def __init__(self):
        super(MixedSignalDataset).__init__()
        with open("./data/train_arr.pkl", "rb") as f:
            self.train_arr = pickle.load(f)
        with open("./data/label_arr.pkl", "rb") as f:
            self.label_arr = pickle.load(f)
        self.keys = list(train_arr.keys())
    
    def __getitem__(self, index):
        cur_key = self.keys[index]
        train_data = train_arr[cur_key]
        label_data = train_arr[cur_key]
        return 