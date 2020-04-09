import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNet(nn.Module):
    """
        model built with 2 lstm layers
    """

    def __init__(self, hidden_size, num_layers, batch_size, output_size):
        super(LSTMNet, self).__init__()
        #! pytorch document: https://pytorch.org/docs/stable/nn.html
        # initilize h0, c0 (num_layers * num_directions, batch, hidden_size)
        # output: (seq_len, batch, num_directions * hidden_size) supposedly, but here we use "batch_first"
        self.hidden = (torch.randn(num_layers, batch_size, hidden_size).cuda(), \
                       torch.randn(num_layers, batch_size, hidden_size).cuda())
        self.double_lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, 
                                   num_layers=2, batch_first=True)
        # batch_first, (batch, seq_len, num_directions * hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.double_lstm(x, self.hidden)
        enc = lstm_out[:, -1]
        linear_out1 = self.linear(enc)
        linear_out = self.output_layer(linear_out1)
        return linear_out

        


