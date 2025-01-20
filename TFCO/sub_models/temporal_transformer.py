# Define the Transformer Model
import torch
import torch.nn as nn
from torch import nn

from utils.transformer_utils import PositionalEncoding


class TemporalTransformer(nn.Module):
    def __init__(self, config, network_configs):
        super(TemporalTransformer, self).__init__()
        feature_size=1024
        num_layers=6
        num_heads=4
        dropout=0.1
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(feature_size, num_heads, feature_size*4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(feature_size, feature_size)

    def forward(self, input_sequence):
        src = self.pos_encoder(input_sequence)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output[-1]

    def _generate_square_subsequent_mask(self, sz): # not needed for now
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

