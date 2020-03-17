import _pickle as pickle
data_dir = "./data/RML2016.10a_dict.pkl"
with open(data_dir, "rb") as f:
    data = pickle.load(f, encoding = "iso-8859-1")

import numpy as np
mod_list = list({key[0] for key in data.keys()})
SNR_range = range(-6, 14, 2)
feature_arr = []
label_arr = []

from tqdm import trange

for state in trange(1<<11):
    print(state)
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
    if (cnt != 5): continue
    label = np.repeat(label, 1000, axis=0)
    print(f"combining {idx_list}")
    for snr in SNR_range:
        cur_feature_arr = np.zeros((1000, 2, 128))
        label_arr.append(label)
        for idx in idx_list:
            mod = mod_list[idx]
            cur_feature_arr += data[(mod, snr)]
        feature_arr.append(cur_feature_arr)

label_arr = np.vstack(label_arr)
feature_arr = np.vstack(feature_arr)

import h5py

with h5py.File("./data/processed_data.h5", "w") as f:
    f.create_dataset('feature_mat', data=feature_arr)
    f.create_dataset('label_mat', data=label_arr)