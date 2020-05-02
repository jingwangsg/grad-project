from utils import Params
from model.cldnn import CLDNN
import torch

params = Params("./experiment/CLDNN/base_model/params.json")
device = torch.device("cuda")
cldnn = CLDNN(params, device)

x = torch.randn((100, 128, 2))

pred = cldnn(x)