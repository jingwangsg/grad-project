import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUNet(nn.Module):
    """
        model built with 2 gru layers
    """

    def __init__(self, params):
        super(GRUNet, self).__init__()
        #! https://pytorch.org/docs/stable/nn.html#torch.nn.GRU 
        # input_size, hidden_size, num_layers
        # h0  (num_layers * num_directions, batch, hidden_size)
        self.h0 = torch.randn((params.num_layers, params.batch_size, params.hidden_size)).cuda()
        self.double_gru = nn.GRU(input_size=params.input_size, \
                                hidden_size=params.hidden_size,\
                                num_layers=params.num_layers,\
                                batch_first=True)
        self.batchnorm = nn.BatchNorm1d(params.hidden_size)
        self.linear_1 = nn.Linear(params.hidden_size, params.hidden_size//2)
        self.linear_2 = nn.Linear(params.hidden_size//2, params.output_size)

    def forward(self, x):
        gru_out, _ = self.double_gru(x, self.h0)
        enc = gru_out[:, -1]
        enc = self.batchnorm(enc)
        enc = self.linear_1(enc)
        linear_out = self.linear_2(enc)
        return linear_out

        


