import torch.nn as nn
from einops import rearrange
from torch import nn


class TemporalConv(nn.Module):
    def __init__(self, config, network_configs):
        super(TemporalConv, self).__init__()
        self.depth = config['sequence_len'] + 1

        self.conv1_seq = nn.Conv2d(in_channels=self.depth, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Norm for conv1

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Norm for conv2

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)  # Batch Norm for conv3

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(10)  # Batch Norm for conv4

        self.conv5 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        s, b, c, h, w = x.shape
        x = rearrange(x, 's b c h w -> b s c h w')
        x = x.reshape(b, s*c, h, w)
        x = self.relu(self.bn1(self.conv1_seq(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)

        # map output to values between 0 and 1
        return x