import torch
import torch.nn as nn
import torch.nn.functional as F

class CLDNN(nn.Module):

    def __init__(self, params, device):
        """ params:
            - num_filters
            - kernel_size
            - pad_size
            - time_step
            - batch_size
            - output_size
        """
        super(CLDNN, self).__init__()
        self.pad1 = nn.ZeroPad2d((params.pad_size, params.pad_size, 0, 0))
        self.conv1 = nn.Conv2d(1, params.num_filters,\
                                kernel_size=(1, params.kernel_size))
        self.pad2 = nn.ZeroPad2d((params.pad_size, params.pad_size, 0, 0))
        self.conv2 = nn.Conv2d(params.num_filters, params.num_filters,\
                               kernel_size=(1, params.kernel_size))
        self.pad3 = nn.ZeroPad2d((params.pad_size, params.pad_size, 0, 0))
        self.conv3 = nn.Conv2d(params.num_filters, params.num_filters,\
                               kernel_size=(1, params.kernel_size))
        # calculate the input dimension of LSTM
        lstm_input_size = params.num_filters * 2
        self.lstm_hidden = (torch.randn(1, params.batch_size, params.lstm_hidden_size).to(device), \
                            torch.randn(1, params.batch_size, params.lstm_hidden_size).to(device))
        self.lstm = nn.LSTM(lstm_input_size, params.lstm_hidden_size, batch_first=True)
        self.linear = nn.Linear(params.lstm_hidden_size, params.output_size)

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        x = x.unsqueeze(1)
        x = x.permute([0, 1, 3, 2])
        x = self.pad1(x)
        x1 = self.conv1(x)
        x2 = self.pad2(x1)
        x2 = self.conv2(x2)
        x3 = self.pad3(x2)
        x3 = self.conv3(x3)
        #(batch, num_filters ,2, x1.shape[-1] + x3.shape[-1])
        x_concat = torch.cat([x1, x3], dim=3)
        x_concat = x_concat.permute([0, 3, 1, 2])
        lstm_input_size = x_concat.shape[-2] * x_concat.shape[-1]
        num_time_step = x_concat.shape[1]
        x_concat = x_concat.reshape((-1, num_time_step, lstm_input_size))
        lstm_out = self.lstm(x_concat)
        linear_out = self.linear(lstm_out)

        return linear_out
        
