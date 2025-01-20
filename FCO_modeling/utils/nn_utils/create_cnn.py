import torch
from torchvision import models, transforms
from torch import nn


# Define the new ResNet model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CustomResNet(nn.Module):
    def __init__(self, sigmoid: bool = False):
        super(CustomResNet, self).__init__()

        self.sigmoid = sigmoid

        # Load the pre-trained ResNet34 model
        self.resnet = models.resnet34(pretrained=True)

        # Modify the first convolutional layer to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )

        # Initialize the new conv1 weights using the mean of the original weights
        with torch.no_grad():
            original_conv1_weight = self.resnet.conv1.weight.clone()
            self.resnet.conv1.weight.copy_(
                original_conv1_weight.mean(dim=1, keepdim=True)
            )

        # Modify the fully connected layer to output 64 features
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 64)

        # Additional layers after concatenating with vector input 'v'
        self.fc1 = nn.Linear(64 + 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Pass the input through the ResNet backbone
        x = self.resnet(x)
        x = F.relu(x)

        # Ensure 'v' is a 2D tensor with shape [batch_size, 2]
        if v.dim() == 1:
            v = v.unsqueeze(1)

        # Concatenate with additional vector input 'v' along the feature dimension
        x = torch.cat((x, v), dim=1)

        # Pass through additional fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        if self.sigmoid:
            x = torch.sigmoid(x)
        return x



if __name__ == '__main__':
    model = CustomResNet()
    out = model.forward(torch.rand(1, 3, 400, 400), torch.rand(1, 2))
    print(out)
