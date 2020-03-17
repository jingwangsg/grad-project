import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MultiCLDNN

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = MultiCLDNN(kernel_size=8).to(device)
