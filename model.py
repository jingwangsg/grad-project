import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, activation="relu"):
        super(ConvBlock, self).__init__()
        self.pad = nn.ZeroPad2d((0, 0, kernel_size//2, kernel_size//2))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size))
        if (activation == "relu"):
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU();
        self.drop = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.activation(x)
        x = self.drop(x)
        return x

class MultiCLDNN(nn.Module):

    def __init__(self, kernel_size, dropout_rate=0.5):
        super(MultiCLDNN, self).__init__()
        self.conv1 = ConvBlock(1, 50, kernel_size, 0.5)
        self.conv2 = ConvBlock(50, 50, kernel_size, 0.5)
        self.conv3 = ConvBlock(50, 50, kernel_size, 0.5)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat([x1, x3])
        batch_size = x4.shape[0]
        time_step = x4.shape[1]
        x4 = x4.reshape((batch_size, time_step, -1))
        input_dim = x4.shape[1]
        lstm_out = nn.LSTM(input_size=input_dim, hidden_size=50, batch_first=True)(x4)
        lstm_out = nn.Sigmoid(lstm_out)
        return lstm_out
