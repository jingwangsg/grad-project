import _pickle as pickle
import h5py
import numpy as np
import argparse
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from tqdm import trange

data_dir = "./data/RML2016.10a_dict.pkl"
with open(data_dir, "rb") as f:
    data = pickle.load(f, encoding = "iso-8859-1")

mod_list = list({key[0] for key in data.keys()})
SNR_range = range(0, 12, 2)
feature_arr = []
label_arr = []

parser = argparse.ArgumentParser()
parser.add_argument("--num_sample")
parser.add_argument("--num_type")
args = parser.parse_args()

sample_per_pair = int(args.num_sample)
num_type = int(args.num_type)


for state in trange(1<<11):
    idx_list = []
    label = np.zeros((1,11), dtype=np.int)
    name = ""
    cnt = 0
    for j in range(11):
        if (state == 0): break
        state >>= 1
        if (state & 1):
            cnt += 1
            idx_list.append(j)
            label[0][j] = 1
            name += "_" + mod_list[j]
    if (cnt != num_type): continue
    label = np.repeat(label, sample_per_pair, axis=0)
    for snr in SNR_range:
        cur_feature_arr = np.zeros((sample_per_pair, 2, 128))
        cur_label = label.copy()
        cur_label = np.concatenate((cur_label, np.ones((sample_per_pair, 1), dtype=np.float)), axis=1)
        label_arr.append(cur_label)
        for idx in idx_list:
            mod = mod_list[idx]
            cur_feature_arr += data[(mod, snr)][:sample_per_pair]
        feature_arr.append(cur_feature_arr)
feature_arr = np.vstack(feature_arr)
feature_arr = feature_arr.reshape((feature_arr.shape[0], -1))
feature_arr = normalize(feature_arr, norm="l2")
feature_arr = feature_arr.reshape((feature_arr.shape[0], 2, 128))
#! https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html
feature_arr = feature_arr.transpose([0, 2, 1])

with h5py.File(f"./data/rml_new_data_{num_type}_{sample_per_pair}_with_high_snr.h5", "w") as f:
    f.create_dataset('feature_mat', data=feature_arr)
    f.create_dataset("label_info", data=label_arr)