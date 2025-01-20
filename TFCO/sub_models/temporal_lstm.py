import torch
import torch.nn as nn
from torch import nn


class TemporalLSTM(nn.Module):
    def __init__(self, config, network_configs):
        super(TemporalLSTM, self).__init__()
        input_dim = 1024
        hidden_dim = 1024
        output_dim = 1024
        num_layers = 2
        self.hidden_dim = hidden_dim

        # Define the LSTM layer with batch_first=False
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(1), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.lstm.num_layers, x.size(1), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.linear(out[-1, :, :])
        return out