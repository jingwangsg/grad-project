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
SNR_range = range(-12, 12, 2)
feature_dict = {}
logit_dict = {}

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
        if (snr not in feature_dict):
            feature_dict[snr] = []
            logit_dict[snr] = []
        cur_feature_arr = np.zeros((sample_per_pair, 2, 128))
        logit_dict[snr].append(label)
        for idx in idx_list:
            mod = mod_list[idx]
            cur_feature_arr += data[(mod, snr)][:sample_per_pair]
        cur_feature_arr = cur_feature_arr.transpose([0, 2, 1])
        feature_dict[snr].append(cur_feature_arr)

for snr in SNR_range:
    feature_dict[snr] = np.vstack(feature_dict[snr])
    logit_dict[snr] = np.vstack(logit_dict[snr])

# import ipdb; ipdb.set_trace()

with open(f"./data/test_data_{num_type}_{sample_per_pair}.h5", "wb") as f:
    pickle.dump((feature_dict, logit_dict), f, protocol=4);
# with h5py.File(f"./data/test_data_{num_type}_{sample_per_pair}.h5", "w") as f:
#     f.create_dataset("logit_dict", data=logit_dict)
#     f.create_dataset('feature_dict', data=feature_dict)
