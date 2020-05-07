import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCellNet(nn.Module):
    """LSTMNet implemented using LSTMCell so that we can use BPTT
    params:
        * input_size - number of dimension of input representation
        * num_layers - number of LSTM layers
        * hidden_size - number of hidden units in LSTM
        * output_size - number of classified category
    """
    def forward(self, x):
        # input - (batch, input_size)
        x = x.permute([1, 0, 2])
        state_list = []
        output_list = []
        for lstm_cell in self.lstm:
            for one_step_x in x:
                if (len(state_list) != 0): state = state_list[-1]
                output, state = lstm_cell(one_step_x)
                output_list.append(output)
                state_list.append(state)
        


