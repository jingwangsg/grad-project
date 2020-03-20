import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, activation="relu"):
        super(ConvBlock, self).__init__()
        self.pad = nn.ZeroPad2d((kernel_size//2, kernel_size//2, 0, 0))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size))
        if (activation == "relu"):
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU();
        self.dropout_rate =dropout_rate

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.activation(x)
        # use upper part to control the dropout function
        x = F.dropout(x, training=self.training, p = self.dropout_rate)
        return x

class MultiCLDNN(nn.Module):

    def __init__(self, kernel_size, dropout_rate=0.5):
        super(MultiCLDNN, self).__init__()
        self.conv1 = ConvBlock(1, 50, kernel_size, 0.5)
        self.conv2 = ConvBlock(50, 50, kernel_size, 0.5)
        self.conv3 = ConvBlock(50, 50, kernel_size, 0.5)
        self.training = False

    def forward(self, x):
        # <type: tuple>: (batch, channel, height, width)
        #conv1, conv2, conv3 include dropout part
        self.conv1.training = self.training
        self.conv2.training = self.training
        self.conv3.training = self.training
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x1 = x1.reshape((x1.shape[0], x1.shape[1], -1))
        x3 = x3.reshape((x3.shape[0], x3.shape[1], -1))
        x4 = torch.cat([x1, x3], dim=2)
        batch_size = x4.shape[0]
        time_step = x4.shape[1]
        x4 = x4.reshape((batch_size, time_step, -1))
        input_dim = x4.shape[1]
        lstm_out = nn.LSTM(input_size=input_dim, hidden_size=50, batch_first=True)(x4)
        lstm_out = nn.Sigmoid(lstm_out)
        return lstm_out
