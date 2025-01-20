import torch
import torch.nn as nn
import torch.nn.functional as F

class RawTrafficEncoder(nn.Module):
    """
    This is a Transformer-based encoder that takes the raw traffic data (e.g., position and angle of vehicles) as input.
    """
    def __init__(self, num_features: int = 3, vehicle_embed_dim: int = 64, num_encoder_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super(RawTrafficEncoder, self).__init__()

        # Vehicle feature embedding network
        self.vehicle_embedding = nn.Sequential(
            nn.Linear(num_features, vehicle_embed_dim),
            nn.ReLU(),
            nn.Linear(vehicle_embed_dim, vehicle_embed_dim),
            nn.ReLU()
        )

        # Additionally provide a CLS token (this is also helpful when there are no vehicles in the scene)
        self.cls_token = nn.Parameter(torch.randn(1, 1, vehicle_embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=vehicle_embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, max_num_vehicles, num_features)

        Returns:
            Tensor: Encoded representation of shape (batch_size, vehicle_embed_dim)
        """
        batch_size, max_num_vehicles, num_features = x.shape

        # Compute padding mask (True for padding positions) --> ensures that attention is only applied to non zero inputs
        padding_mask = (x.abs().sum(dim=2) == 0)  # Shape: (batch_size, max_num_vehicles)

        # Pass through vehicle embedding network --> create embedding tokens for each vehicle (even if they are zero)
        x_embedded = self.vehicle_embedding(x)  # Shape: (batch_size, max_num_vehicles, vehicle_embed_dim)

        # Expand cls_token to match batch size
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, vehicle_embed_dim)

        # Concatenate cls_token with the embedded vehicle features
        x_embedded = torch.cat([cls_token, x_embedded], dim=1)  # Shape: (batch_size, 1 + max_num_vehicles, vehicle_embed_dim)

        # Update padding mask to include cls_token (which should not be masked)
        cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        padding_mask = torch.cat([cls_padding, padding_mask], dim=1)  # Shape: (batch_size, 1 + max_num_vehicles)

        # Transpose for transformer encoder (sequence_length, batch_size, embedding_dim)
        x_embedded = x_embedded.transpose(0, 1)  # Shape: (1 + max_num_vehicles, batch_size, vehicle_embed_dim)

        # Apply transformer encoder
        x_encoded = self.transformer_encoder(x_embedded, src_key_padding_mask=padding_mask)
        # Shape: (sequence_length, batch_size, vehicle_embed_dim)

        # Transpose back to (batch_size, sequence_length, vehicle_embed_dim)
        x_encoded = x_encoded.transpose(0, 1)  # Shape: (batch_size, 1 + max_num_vehicles, vehicle_embed_dim)

        # Extract the cls_token representation
        cls_token = x_encoded[:, 0, :]

        return cls_token

if __name__ == "__main__":
    # Parameters
    batch_size = 32
    max_num_vehicles = 100
    num_features = 3  # For position (x, y) and angle

    # Create a dummy input tensor
    tensor = torch.zeros(batch_size, max_num_vehicles, num_features)
    # Simulate data for the first 5 vehicles in each sequence
    tensor[:, 0:5, :] = torch.rand(batch_size, 5, num_features)

    # Initialize the encoder
    encoder = RawTrafficEncoder(num_features=num_features, max_num_vehicles=max_num_vehicles)

    # Forward pass
    output = encoder(tensor)
    print("Output shape:", output.shape)  # Should be (batch_size, sequence_length, time_embed_dim)
