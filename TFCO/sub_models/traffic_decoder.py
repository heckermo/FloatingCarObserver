import torch
import torch.nn as nn

class TrafficDecoder(nn.Module):
    """
    Transformer-based decoder that takes the processed embeddings of the temporal encoder
    as input and outputs N position (x, y) predictions for each vehicle.
    Inspired by the DETR architecture (https://arxiv.org/abs/2005.12872).
    """
    def __init__(
        self,
        num_queries: int = 100, # Maximum number of vehicles to predict
        hidden_dim: int = 64,
        num_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super(TrafficDecoder, self).__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
        )
        # Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Final linear layer to map decoder outputs to position predictions
        self.output_layer = nn.Linear(hidden_dim, 3)  # Predicts (no_class, x, y)

    def forward(self, encoder_outputs):
        """
        Forward pass for the decoder.

        Args:
            encoder_outputs (torch.Tensor): Output from the encoder, shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]

        Returns:
            torch.Tensor: Position predictions, shape [batch_size, num_queries, 2].
        """
        # Ensure that the encoder output has the correct shape
        if len(encoder_outputs.shape) == 2:
            encoder_outputs = encoder_outputs.unsqueeze(1)

        # bring seq_len to the first dimension
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs: [seq_len, batch_size, hidden_dim]
        seq_len, batch_size, hidden_dim = encoder_outputs.shape

        # Prepare query embeddings
        # query_embed: [num_queries, batch_size, hidden_dim]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Decode the sequence
        # tgt: target sequence (query embeddings)
        # memory: encoder outputs
        decoder_outputs = self.transformer_decoder(
            tgt=query_embed,
            memory=encoder_outputs,
        )  # [num_queries, batch_size, hidden_dim]

        # Predict positions
        # positions: [num_queries, batch_size, 3]
        positions = self.output_layer(decoder_outputs)

        # Transpose to shape [batch_size, num_queries, 3]
        positions = positions.permute(1, 0, 2)

        return positions

if __name__ == "__main__":
    # Parameters
    batch_size = 32
    seq_len = 100
    hidden_dim = 256

    # Create a dummy input tensor
    tensor = torch.rand(batch_size, seq_len, hidden_dim)

    # Initialize the decoder
    decoder = TrafficDecoder(num_queries=100, hidden_dim=hidden_dim)

    # Forward pass
    output = decoder(tensor)
    print(output.shape)
