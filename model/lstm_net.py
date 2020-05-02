import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNet(nn.Module):
    """
        model built with 2 lstm layers
    """

    def __init__(self, params, device):
        super(LSTMNet, self).__init__()
        #! https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        # initilize h0, c0 (num_layers * num_directions, batch, hidden_size)
        # output: (seq_len, batch, num_directions * hidden_size) supposedly, but here we use "batch_first"

        self.hidden = (torch.randn(params.num_layers, params.batch_size, params.hidden_size).to(device), \
                       torch.randn(params.num_layers, params.batch_size, params.hidden_size).to(device))
        self.double_lstm = nn.LSTM(input_size=params.input_size, hidden_size=params.hidden_size, 
                                   num_layers=params.num_layers, batch_first=True)
        # batch_first, (batch, seq_len, num_directions * hidden_size)
        #! https://pytorch.org/docs/stable/nn.html#batchnorm1d
        #  C from an expected input of size (N, C, L) or L from input of size (N, L)
        self.batchnorm = nn.BatchNorm1d(params.hidden_size)
        self.linear1 = nn.Linear(params.hidden_size, params.hidden_size//2)
        self.selu = nn.SELU()
        self.linear2 = nn.Linear(params.hidden_size//2, params.output_size)

    def forward(self, x):
        lstm_out, _ = self.double_lstm(x, self.hidden)
        enc = lstm_out[:, -1]
        enc = self.batchnorm(enc)
        enc = self.linear1(enc)
        enc = self.selu(enc)
        linear_out = self.linear2(enc)
        return linear_out

        


