import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        self.attention_weights = attn.detach()
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, context):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)
        k, v = kv

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b m (h d) -> b h m d', h = self.heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ViTEncoderDecoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, decoder_depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., sigmoid_activation: bool = True, vector_dim=2):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Encoder
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Vector embedding
        self.vector_embedding = nn.Sequential(
            nn.Linear(vector_dim, dim),
            nn.LayerNorm(dim)
        )

        # Decoder
        self.decoder = TransformerDecoder(dim, decoder_depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.sigmoid = sigmoid_activation

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        if self.sigmoid:
            self.mlp_head.add_module('sigmoid', nn.Sigmoid())

    def forward(self, img, vector):
        # Patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Encoder
        encoder_output = self.encoder(x)  # Shape: (b, n+1, dim)

        # Embed the vector
        vector_embedded = self.vector_embedding(vector).unsqueeze(1)  # Shape: (b, 1, dim)

        # Decoder
        decoder_output = self.decoder(vector_embedded, context=encoder_output)

        # Use the decoder output for classification
        decoder_output = decoder_output.squeeze(1)  # Shape: (b, dim)

        decoder_output = self.to_latent(decoder_output)
        logits = self.mlp_head(decoder_output)
        return logits

    def get_last_selfattention(self):
        # Retrieve the attention weights from the encoder layers
        attentions = [layer[0].fn.attention_weights for layer in self.encoder.layers]
        return torch.stack(attentions)

if __name__ == "__main__":
    # Define model parameters
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1,
        dim=64,            # Model dimension
        depth=3,            # Encoder depth
        decoder_depth=2,    # Decoder depth
        heads=4,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        channels=1,         # Set to 1 if using grayscale images
        sigmoid_activation=True,
        vector_dim=2        # Dimension of the additional vector
    )

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Get the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Example inputs
    img = torch.randn(8, 1, 224, 224).to(device)  # Batch of grayscale images
    vector = torch.randn(8, 2).to(device)         # Batch of vectors

    # Forward pass
    t = time.time()
    outputs = model(img, vector)
    print("Time taken for forward pass:", time.time() - t)

    print("Output shape:", outputs.shape)