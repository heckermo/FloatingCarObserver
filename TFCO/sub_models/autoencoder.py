import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.encoder_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # output: [16, 200, 200] (400)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # output: [32, 100, 100] (400)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # output: [64, 50, 50] or [32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # output: [128, 25, 25]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 1024)
        )

    def forward(self, x):
        return self.encoder_layers(x)


class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.decoder_layers = nn.Sequential(
            nn.Linear(1024, 32*32*32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 32, 32)),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            #nn.Sigmoid()  # Use Sigmoid for the final layer if the input is normalized between 0 and 1
        )

    def forward(self, x):
        return self.decoder_layers(x)
    

