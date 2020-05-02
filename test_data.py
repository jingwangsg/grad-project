import h5py
import torch

with h5py.File("./data/rml_data_5_10_with_high_snr.h5", "r") as f:
    feature_mat = torch.Tensor(f["feature_mat"])
    logit_mat = torch.Tensor(f["logit_mat"])

import ipdb; ipdb.set_trace()