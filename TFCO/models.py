import logging

import torch
import torch.nn as nn
from einops import rearrange

from sub_models.autoencoder import CNNDecoder, CNNEncoder
from sub_models.conv_lstm import ConvLSTM
from sub_models.d3conv import D3CNN
from sub_models.raw_traffic_encoder import RawTrafficEncoder
from sub_models.traffic_decoder import TrafficDecoder
from sub_models.temporal_conv import TemporalConv
from sub_models.temporal_lstm import TemporalLSTM
from sub_models.temporal_transformer import TemporalTransformer
from sub_models.vit_encoder import init_ViT

from typing import Dict

logger = logging.getLogger(__name__)

SUBNETWORKS = {
    'Encoder': {
        'CNN_Encoder': CNNEncoder,
        'VIT': init_ViT,
        'RawTrafficEncoder': RawTrafficEncoder,
    },
    'TemporalModule': {
        'Transformer': {"model_class": TemporalTransformer, "encoder_active": True, "decoder_active": True},
        'Conv': {"model_class": TemporalConv, "encoder_active": False, "decoder_active": False},
        'LSTM': {"model_class": TemporalLSTM, "encoder_active": True, "decoder_active": True},
        '3DCNN': {"model_class": D3CNN, "encoder_active": False, "decoder_active": False},
        'ConvLSTM': {"model_class": ConvLSTM, "encoder_active": False, "decoder_active": False},
    },
    'Decoder': {
        'CNN_Decoder': CNNDecoder,
        'Position_Decoder': TrafficDecoder,
    }
}

class SpacialTemporalDecoder(nn.Module):
    def __init__(self, config: dict, network_configs: dict):
        super().__init__()
        self.config = config
        self.network_configs = network_configs
        self._initialize_attributes()
        self._initialize_modules()
        self._load_pretrained()
        self._fix_subnetworks()
        self._check_validity()

    def _initialize_attributes(self):
        self.input_type = self.config.get('input_type', 'bev')
        self.output_type = self.config.get('output_type', 'bev')
        self.max_vehicles = self.config.get('max_vehicles', 100)
        self.image_size = self.config.get('image_size', 256)
        self.pre_train = self.config.get('pre_train', False)
        self.sequence_len = self.config.get('sequence_len', 1)
        self.res_con = self.config.get('residual_connection', False)
        self.sigmoid_factor = self.config.get('sigmoid_factor', 1.0)
        self.temporal_active = self.config.get('temporal') is not None

    def _initialize_modules(self):
        temporal_config = self.config.get('temporal')
        if temporal_config:
            temporal_info = SUBNETWORKS['TemporalModule'][temporal_config]
            self.temporal_module = temporal_info['model_class'](self.config, self.network_configs)
            self.encoder_active = temporal_info.get('encoder_active', True)
            self.decoder_active = temporal_info.get('decoder_active', True)
        else:
            self.temporal_module = None
            self.encoder_active = True
            self.decoder_active = True

        if self.encoder_active:
            encoder_class = SUBNETWORKS['Encoder'][self.config['encoder']]
            self.encoder = encoder_class()

        if self.decoder_active:
            decoder_class = SUBNETWORKS['Decoder'][self.config['decoder']]
            self.decoder = decoder_class()

    def forward(self, x: torch.Tensor, batch_first: bool = True) -> torch.Tensor:
        if batch_first:
            x = x.transpose(0, 1)  # Swap sequence and batch dimensions

        # Encode the input tensor
        # x is of shape (seq_len, batch_size, -1)
        x = self._encode(x)

        # Apply the temporal module if active
        # x is of shape (seq_len, batch_size, embed_dim)
        if self.temporal_active and self.temporal_module is not None:
            x = self.temporal_module(x)
        else:
            x = x.squeeze(dim=0 if self.sequence_len == 1 else 1)
        
        # Decode the tensor if the decoder is active
        # x is of shape (batch_size, embed_dim)
        if self.decoder_active and self.decoder is not None:
            x = self.decoder(x)

        # Reshape the output tensor to the desired format
        # x is of shape (batch_size, -1)
        x = self._reshape_output(x)

        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.encoder_active or self.encoder is None:
            return x

        if self.input_type == 'bev':
            seq_len, batch_size, channels, height, width = x.shape
            x = x.reshape(seq_len * batch_size, channels, height, width)
            x = self.encoder(x)
            x = x.reshape(seq_len, batch_size, -1)
        elif self.input_type == 'raw':
            seq_len, batch_size, num_vehicles, num_features = x.shape
            x = x.reshape(seq_len * batch_size, num_vehicles, num_features)
            x = self.encoder(x)
            x = x.reshape(seq_len, batch_size, -1)
        return x

    def _reshape_output(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self.output_type == 'bev':
            channels, height, width = self.image_size
            x = x.reshape(batch_size, channels, height, width)
        elif self.output_type == 'raw':
            x = x.reshape(batch_size, self.max_vehicles, -1)
        return x

    def _load_pretrained(self):
        if self.config.get('load_encoder') and self.encoder:
            self.encoder.load_state_dict(torch.load(self.config['load_encoder']), strict=False)
            logger.info('Loaded encoder weights.')

        if self.config.get('load_temporal') and self.temporal_module:
            self.temporal_module.load_state_dict(torch.load(self.config['load_temporal']), strict=False)
            logger.info('Loaded temporal module weights.')

        if self.config.get('load_decoder') and self.decoder:
            self.decoder.load_state_dict(torch.load(self.config['load_decoder']), strict=False)
            logger.info('Loaded decoder weights.')

    def _fix_subnetworks(self):
        if self.config.get('fix_encoder') and self.encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.config.get('fix_temporal') and self.temporal_module:
            for param in self.temporal_module.parameters():
                param.requires_grad = False

        if self.config.get('fix_decoder') and self.decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
    
    def _check_validity(self):
        if self.input_type == 'raw':
            if not (self.encoder_active and self.config['encoder'] == "RawTrafficEncoder"):
                raise ValueError("Raw traffic input requires the RawTrafficEncoder.")
            if self.config.get('temporal') != 'Transformer':
                raise ValueError("Raw traffic input requires the Transformer temporal module.")

        if self.output_type == 'raw':
            if not (self.decoder_active and self.config['decoder'] == "Position_Decoder"):
                raise ValueError("Raw traffic output requires the Position_Decoder.")
            if self.config.get('temporal') != 'Transformer':
                raise ValueError("Raw traffic output requires the Transformer temporal module.")

        if self.config.get('encoder') == "RawTrafficEncoder" and self.input_type != 'raw':
            raise ValueError("The RawTrafficEncoder can only be used with raw input.")

        if self.config.get('decoder') == "Position_Decoder" and self.output_type != 'raw':
            raise ValueError("The Position_Decoder can only be used with raw output.")

        temporal_config = self.config.get('temporal')
        if temporal_config:
            temporal_info = SUBNETWORKS['TemporalModule'][temporal_config]
            if not temporal_info.get('encoder_active', True) and self.config.get('encoder'):
                logger.warning(f"Encoder is not used with {temporal_config}. Setting encoder to None.")
                self.config['encoder'] = None

            if not temporal_info.get('decoder_active', True) and self.config.get('decoder'):
                logger.warning(f"Decoder is not used with {temporal_config}. Setting decoder to None.")
                self.config['decoder'] = None

 

class SpacialDecoder(nn.Module):
    """
    This class is used to create a n encoder-decoder architecture used to pre-train those networks before using them in the SpacialTemporalDecoder.
    I.e. when the model is pre-trained
    """
    def __init__(self, config: dict, network_configs: dict):
        super().__init__()
        self.config = config
        self.network_configs = network_configs
        self._initialize_attributes()
        self._check_validity()
        self._initialize_modules()
        self._load_pretrained()
        self._fix_subnetworks()
        
    
    def _initialize_attributes(self):
        self.input_type = self.config.get('input_type', 'bev')
        self.output_type = self.config.get('output_type', 'bev')
        self.max_vehicles = self.config.get('max_vehicles')
        self.image_size = self.config.get('image_size')
        self.sigmoid_factor = self.config.get('sigmoid_factor', 1.0)

    def _initialize_modules(self):
        encoder_class = SUBNETWORKS['Encoder'][self.config['encoder']]
        self.encoder = encoder_class()

        decoder_class = SUBNETWORKS['Decoder'][self.config['decoder']]
        self.decoder = decoder_class()

    def forward(self, x: torch.Tensor, batch_first: bool = True) -> torch.Tensor:
        if batch_first:
            x = x.transpose(0, 1)  # Swap sequence and batch dimensions
        
        # Encode the input tensor
        # x will be of shape (1, batch_size, -1) (seq_len = 1 for the pre-training)
        x = self._encode(x)

        # Remove the sequence dimension (would be done by the temporal module in the SpacialTemporalDecoder)
        # x will be of shape (batch_size, embed_dim)
        x = x.squeeze(dim=0)

        # Decode the tensor
        # x will be of shape (batch_size, embed_dim)
        x = self.decoder(x)

        # Reshape the output tensor to the desired format
        # x will be of shape (batch_size, -1)
        x = self._reshape_output(x)

        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_type == 'bev':
            seq_len, batch_size, channels, height, width = x.shape
            x = x.reshape(seq_len * batch_size, channels, height, width)
            x = self.encoder(x)
            x = x.reshape(seq_len, batch_size, -1)
        elif self.input_type == 'raw':
            seq_len, batch_size, num_vehicles, num_features = x.shape
            x = x.reshape(seq_len * batch_size, num_vehicles, num_features)
            x = self.encoder(x)
            x = x.reshape(seq_len, batch_size, -1)
        return x

    def _reshape_output(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self.output_type == 'bev':
            channels, height, width = self.image_size
            x = x.reshape(batch_size, channels, height, width)
        elif self.output_type == 'raw':
            x = x.reshape(batch_size, self.max_vehicles, -1)
        return x

    def _load_pretrained(self):
        if self.config.get('load_encoder') and self.encoder:
            self.encoder.load_state_dict(torch.load(self.config['load_encoder']), strict=False)
            logger.info('Loaded encoder weights.')

        if self.config.get('load_decoder') and self.decoder:
            self.decoder.load_state_dict(torch.load(self.config['load_decoder']), strict=False)
            logger.info('Loaded decoder weights.')

    def _fix_subnetworks(self):
        if self.config.get('fix_encoder') and self.encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.config.get('fix_decoder') and self.decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False


    def _check_validity(self):
        if self.config['temporal'] is not None:
            logger.warning("In the pre-training mode, the temporal module is not used.")
            self.config['temporal'] = None
        
        if self.config['encoder'] is None or self.config['decoder'] is None:
            raise ValueError("Pre-training is designed for encoder-decoder networks. Please provide encoder and decoder.")

        if self.config.get('encoder') == "RawTrafficEncoder" and self.input_type != 'raw':
            raise ValueError("The RawTrafficEncoder can only be used with raw input.")

        if self.config.get('decoder') == "Position_Decoder" and self.output_type != 'raw':
            raise ValueError("The Position_Decoder can only be used with raw output.")

class MaskedSequenceTransformer(nn.Module):
    """
    A transformer-based model for processing masked sequences with attention mechanisms.

    Args:
        sequence_len (int): Length of the input sequence.  # Number of sequence elements to process.
        max_vehicles (int): Maximum number of vehicles represented in the sequence.  # Constraint for sequence padding.
        full_attention (bool): Whether to use full attention across all elements.  # True for full attention, False otherwise.
        vehicle_only_attention (bool): Whether to restrict attention to vehicles only the same vehicles in different time steps.  
        embed_dim (int): Dimensionality of the embedding space.  # Size of embeddings per token.
        num_heads (int): Number of attention heads in the transformer.  # Enables multi-head attention.
        num_layers (int): Number of transformer layers.  # Determines the depth of the transformer.
        dropout (float): Dropout probability for regularization.  # Prevents overfitting.
        num_encoder_layers (int): Number of encoder layers.  # Layers in the encoder stack.
    """
    def __init__(self, sequence_len: int, max_vehicles: int, full_attention: bool = False, vehicle_only_attention: bool = False,
                 embed_dim: int = 64, num_heads: int = 8, dropout: float = 0.1, num_layers: int = 4,
                 use_pos_embed: bool = True, use_type_embed: bool = True):
        super(MaskedSequenceTransformer, self).__init__()
        
        self.sequence_len = sequence_len
        self.max_vehicles = max_vehicles
        self.full_attention = full_attention
        self.vehicle_only_attention = vehicle_only_attention
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_pos_embed = use_pos_embed
        self.use_type_embed = use_type_embed

        # Corrected embed_dim reference
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.num_heads, dropout=self.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Embedding layers
        self.raw_traffic_embedder = nn.Linear(3, self.embed_dim)

        # Positional and type embeddings
        self.pos_embedding = nn.Embedding(self.sequence_len, self.embed_dim)
        self.type_embedding = nn.Embedding(self.max_vehicles, self.embed_dim)

        self.prediction_head = nn.Linear(self.embed_dim, 3)
    
    def forward(self, input_sequence: torch.Tensor):
        batch_size, sequence_len, max_vehicles, num_features = input_sequence.shape
        assert sequence_len == self.sequence_len
        assert max_vehicles == self.max_vehicles
        assert num_features == 3

        # Create a mask for zero inputs to avoid attending to zero inputs
        zero_mask = (input_sequence.sum(dim=-1) == 0)  # Shape: (batch_size, sequence_len, max_vehicles)
        
        # Embed the raw traffic data
        input_sequence = self.raw_traffic_embedder(input_sequence)  # (batch_size, sequence_len, max_vehicles, embed_dim)
        
        # Flatten the sequence for the transformer
        flattened_sequence = rearrange(input_sequence, 'b s v e -> b (s v) e')  # (batch_size, sequence_len * max_vehicles, embed_dim)
        
        # Create time step indices for each position in the flattened sequence
        time_step_indices = torch.arange(sequence_len).unsqueeze(1).repeat(1, max_vehicles).view(-1).to(input_sequence.device)
        time_step_indices = time_step_indices.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, sequence_len * max_vehicles)
        
        # Create vehicle indices for each position
        vehicle_indices = torch.arange(max_vehicles).unsqueeze(0).repeat(sequence_len, 1).view(-1).to(input_sequence.device)
        vehicle_indices = vehicle_indices.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, sequence_len * max_vehicles)
        
        # Create attention masks
        time_step_diff = time_step_indices.unsqueeze(2) != time_step_indices.unsqueeze(1)
        vehicle_diff = vehicle_indices.unsqueeze(2) != vehicle_indices.unsqueeze(1)
        
        if self.vehicle_only_attention:
            attention_mask = vehicle_diff # (batch_size, seq_len * max_vehicles, seq_len * max_vehicles)
        else:
            # Mask positions where both time steps and vehicle indices are different
            attention_mask = time_step_diff & vehicle_diff  # (batch_size, seq_len * max_vehicles, seq_len * max_vehicles)
        
        # Incorporate the zero mask
        zero_mask_flat = zero_mask.view(batch_size, -1)  # (batch_size, sequence_len * max_vehicles)
        zero_mask_expanded = zero_mask_flat.unsqueeze(1) | zero_mask_flat.unsqueeze(2)

        if self.full_attention:
            attention_mask = zero_mask_expanded
        else:
            attention_mask = attention_mask | zero_mask_expanded  # Combine with existing attention mask

        # Convert the attention mask for the transformer (needs to be float with -inf where masked)
        attention_mask = attention_mask.float()
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float(0.0))

        # Expand attention mask for multiple heads
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = attention_mask.view(batch_size * self.num_heads, sequence_len * max_vehicles, sequence_len * max_vehicles)

        vehicle_indices = torch.arange(self.max_vehicles).unsqueeze(0).repeat(self.sequence_len, 1).view(-1).to(input_sequence.device)
        vehicle_indices = vehicle_indices.unsqueeze(0).repeat(batch_size, 1)

        time_step_indices = torch.arange(self.sequence_len).unsqueeze(1).repeat(1, self.max_vehicles).view(-1).to(input_sequence.device)
        time_step_indices = time_step_indices.unsqueeze(0).repeat(batch_size, 1)  # Shape: (batch_size, sequence_len * max_vehicles)

        # Add positional and type embeddings
        if self.use_pos_embed:  
            pos_embed = self.pos_embedding(time_step_indices)  # (batch_size, sequence_len * max_vehicles, embed_dim)
        else:
            pos_embed = torch.zeros(batch_size, sequence_len * max_vehicles, self.embed_dim).to(input_sequence.device)
        
        if self.use_type_embed:
            type_embed = self.type_embedding(vehicle_indices) # (batch_size, sequence_len * max_vehicles, embed_dim)
        else:
            type_embed = torch.zeros(batch_size, sequence_len * max_vehicles, self.embed_dim).to(input_sequence.device)

        flattened_sequence = flattened_sequence + pos_embed + type_embed

        # Transpose for transformer input (seq_len, batch_size, embed_dim)
        transformer_input = flattened_sequence.transpose(0, 1)  # (seq_len * max_vehicles, batch_size, embed_dim)

        # Apply transformer encoder
        transformer_output = self.transformer_encoder(transformer_input, mask=attention_mask)

        # Transpose back and reshape
        transformer_output = transformer_output.transpose(0, 1)  # (batch_size, seq_len * max_vehicles, embed_dim)
        transformer_output = transformer_output.view(batch_size, sequence_len, max_vehicles, self.embed_dim)

        # Only use the information of the last time step
        transformer_output = transformer_output[:, -1, :, :] #(batch_size, max_vehicles, embed_dim)

        # Apply prediction head
        predictions = self.prediction_head(transformer_output)  # (batch_size, sequence_len, max_vehicles, 3)

        return predictions
    

class MaskedSequenceTransformerOverlap(nn.Module):
    """
    A transformer-based model for processing masked sequences with attention mechanisms.

    Args:
        sequence_len (int): Length of the input sequence.  # Number of sequence elements to process.
        max_vehicles (int): Maximum number of vehicles represented in the sequence.  # Constraint for sequence padding.
        num_grids (int): Number of overlap grids for embedding.
        full_attention (bool): Whether to use full attention across all elements.  # True for full attention, False otherwise.
        vehicle_only_attention (bool): Whether to restrict attention to vehicles only the same vehicles in different time steps.  
        embed_dim (int): Dimensionality of the embedding space.  # Size of embeddings per token.
        num_heads (int): Number of attention heads in the transformer.  # Enables multi-head attention.
        num_layers (int): Number of transformer layers.  # Determines the depth of the transformer.
        dropout (float): Dropout probability for regularization.  # Prevents overfitting.
        num_encoder_layers (int): Number of encoder layers.  # Layers in the encoder stack.
        poi_embedding_dim (int): Dimensionality of POI embedding.
        overlap_embedding_dim (int): Dimensionality of overlap embedding.
        use_pos_embed (bool): Whether to use positional embeddings.
        use_type_embed (bool): Whether to use type embeddings.
    """
    def __init__(self, sequence_len: int, max_vehicles: int, num_grids: int, full_attention: bool = False,
                 vehicle_only_attention: bool = False, embed_dim: int = 64, num_heads: int = 8,
                 dropout: float = 0.1, num_layers: int = 4, poi_embedding_dim: int = 8,
                 overlap_embedding_dim: int = 8, use_pos_embed: bool = True, use_type_embed: bool = True):
        super(MaskedSequenceTransformerOverlap, self).__init__()

        self.sequence_len = sequence_len
        self.max_vehicles = max_vehicles
        self.full_attention = full_attention
        self.vehicle_only_attention = vehicle_only_attention
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_pos_embed = use_pos_embed
        self.use_type_embed = use_type_embed


        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        #Raw traffic embedding 
        self.raw_traffic_embedder = nn.Linear(3, self.embed_dim)

        # Positional and type embeddings
        self.pos_embedding = nn.Embedding(self.sequence_len, self.embed_dim)
        self.type_embedding = nn.Embedding(self.max_vehicles, self.embed_dim)

        # POI embedding -> Every POI gets an ID
        self.poi_embedding = nn.Embedding(num_grids + 1, poi_embedding_dim, padding_idx=0)

        # Overlap embedding -> (vehicle seen in radius before (1) or not (0))
        self.overlap_embedding = nn.Embedding(2 + 1, overlap_embedding_dim, padding_idx=0)

        # Combine all embeddings
        self.combine = nn.Linear(embed_dim + overlap_embedding_dim + poi_embedding_dim, embed_dim)

        # Prediction head to output 3 values per vehicle
        self.prediction_head = nn.Linear(self.embed_dim, 3)

    def forward(self, input_sequence: Dict[str, torch.Tensor]):
        
        features = input_sequence["features"]
        overlap_tag = input_sequence["overlap_tag"]
        poi_ids = input_sequence["poi_id"]

        batch_size, sequence_len, max_vehicles, num_features = features.shape
        assert sequence_len == self.sequence_len
        assert max_vehicles == self.max_vehicles
        assert num_features == 3

        # Create a mask for zero inputs to avoid attending to zero inputs
        zero_mask = (features.sum(dim=-1) == 0)  

        # Prepare embeddings for overlap and POI (shift +1 to = 0 as padding)
        overlap_for_embedding = overlap_tag.clone()
        overlap_for_embedding[overlap_for_embedding < 0] = 0  # -> Padding 0
        
        poi_for_embedding = poi_ids.clone()
        poi_for_embedding[poi_for_embedding < 0] = 0  # -> Padding 0 

        # Embeds raw traffic features, overlap and POI
        features = self.raw_traffic_embedder(features)
        overlap_embedding = self.overlap_embedding(overlap_for_embedding)
        poi_embedding = self.poi_embedding(poi_for_embedding)

        # Concat all embeddings and combine
        concated = torch.cat([features, overlap_embedding, poi_embedding], dim=-1)
        concated = self.combine(concated)

        # Flatten the sequence for the transformer
        flattened_sequence = rearrange(concated, 'b s v e -> b (s v) e')

        # Create time step and vehicle indices
        time_step_indices = torch.arange(sequence_len).unsqueeze(1).repeat(1, max_vehicles).view(-1).to(features.device)
        time_step_indices = time_step_indices.unsqueeze(0).repeat(batch_size, 1)
        vehicle_indices = torch.arange(max_vehicles).unsqueeze(0).repeat(sequence_len, 1).view(-1).to(features.device)
        vehicle_indices = vehicle_indices.unsqueeze(0).repeat(batch_size, 1)

        # Create attention masks
        time_step_diff = time_step_indices.unsqueeze(2) != time_step_indices.unsqueeze(1)
        vehicle_diff = vehicle_indices.unsqueeze(2) != vehicle_indices.unsqueeze(1)
        
        if self.vehicle_only_attention:
            attention_mask = vehicle_diff # (batch_size, seq_len * max_vehicles, seq_len * max_vehicles)
        else:
            # Mask positions where both time steps and vehicle indices are different
            attention_mask = time_step_diff & vehicle_diff  # (batch_size, seq_len * max_vehicles, seq_len * max_vehicles)
        
        # Incorporate the zero mask
        zero_mask_flat = zero_mask.view(batch_size, -1)  # (batch_size, sequence_len * max_vehicles)
        zero_mask_expanded = zero_mask_flat.unsqueeze(1) | zero_mask_flat.unsqueeze(2)

        if self.full_attention:
            attention_mask = zero_mask_expanded
        else:
            attention_mask = attention_mask | zero_mask_expanded  # Combine with existing attention mask

        # Convert the attention mask for the transformer (needs to be float with -inf where masked)
        attention_mask = attention_mask.float()
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float(0.0))

        # Expand attention mask for multiple heads
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = attention_mask.view(batch_size * self.num_heads, sequence_len * max_vehicles, sequence_len * max_vehicles)

        # Add positional and type embeddings
        if self.use_pos_embed:
            pos_embed = self.pos_embedding(time_step_indices)  # (batch_size, sequence_len * max_vehicles, embed_dim)
        else:
            pos_embed = torch.zeros(batch_size, sequence_len * max_vehicles, self.embed_dim).to(features.device)
        
        if self.use_type_embed:
            type_embed = self.type_embedding(vehicle_indices) # (batch_size, sequence_len * max_vehicles, embed_dim)
        else:
            type_embed = torch.zeros(batch_size, sequence_len * max_vehicles, self.embed_dim).to(features.device)

        flattened_sequence = flattened_sequence + pos_embed + type_embed

        # Transpose for transformer input (seq_len, batch_size, embed_dim)
        transformer_input = flattened_sequence.transpose(0, 1)  # (seq_len * max_vehicles, batch_size, embed_dim)

        # Apply transformer encoder
        transformer_output = self.transformer_encoder(transformer_input, mask=attention_mask)

        # Transpose back and reshape
        transformer_output = transformer_output.transpose(0, 1)  # (batch_size, seq_len * max_vehicles, embed_dim)
        transformer_output = transformer_output.view(batch_size, sequence_len, max_vehicles, self.embed_dim)

        # Only use the information of the last time step
        transformer_output = transformer_output[:, -1, :, :] #(batch_size, max_vehicles, embed_dim)

        # Apply prediction head
        predictions = self.prediction_head(transformer_output)  # (batch_size, sequence_len, max_vehicles, 3)

        return predictions


if __name__ == "__main__":   
    raise NotImplementedError("This script is not intended to be run directly.")        