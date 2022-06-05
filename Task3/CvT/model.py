import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange

CvT_config = {
    'Stage-1':
        {
            'input_size': (3, 224, 224),
            'embed_dim': 64,
            'patch_size': 7,
            'patch_stride': 4,
            'patch_padding': 2,
            'hidden_dim': 256,
            'num_heads': 1,
            'num_layers': 1,
            'class_token': False,
            'dropout': 0.1
        },
    'Stage-2':
        {
            'input_size': (64, 56, 56),
            'embed_dim': 192,
            'patch_size': 3,
            'patch_stride': 2,
            'patch_padding': 1,
            'hidden_dim': 768,
            'num_heads': 3,
            'num_layers': 6,
            'class_token': False,
            'dropout': 0.1
        },
    'Stage-3':
        {
            'input_size': (192, 28, 28),
            'embed_dim': 384,
            'patch_size': 3,
            'patch_stride': 2,
            'patch_padding': 1,
            'hidden_dim': 1536,
            'num_heads': 6,
            'num_layers': 24,
            'class_token': True,
            'dropout': 0.1
        },
    'Classifier':
        {
            'embed_dim': 384,
            'num_classes': 100,
            'dropout': 0.1,
        }
}

CvT_config_imagenet = {
    'Stage-1':
        {
            'input_size': (3, 224, 224),
            'embed_dim': 64,
            'patch_size': 7,
            'patch_stride': 4,
            'patch_padding': 2,
            'hidden_dim': 256,
            'num_heads': 1,
            'num_layers': 1,
            'class_token': False,
            'dropout': 0.1
        },
    'Stage-2':
        {
            'input_size': (64, 56, 56),
            'embed_dim': 192,
            'patch_size': 3,
            'patch_stride': 2,
            'patch_padding': 1,
            'hidden_dim': 768,
            'num_heads': 3,
            'num_layers': 6,
            'class_token': False,
            'dropout': 0.1
        },
    'Stage-3':
        {
            'input_size': (192, 28, 28),
            'embed_dim': 384,
            'patch_size': 3,
            'patch_stride': 2,
            'patch_padding': 1,
            'hidden_dim': 1536,
            'num_heads': 6,
            'num_layers': 24,
            'class_token': True,
            'dropout': 0.1
        },
    'Classifier':
        {
            'embed_dim': 384,
            'num_classes': 1000,
            'dropout': 0.1,
        }
}


class ConvTransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, kernel_size_q, kernel_size_kv,
                 stride_q, stride_kv, padding_q, padding_kv, class_token=True, dropout=0.1):
        super(ConvTransformerBlock, self).__init__()

        self.class_token = class_token

        # Convolutional Projection
        self.conv_q = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size_q, stride_q, padding_q, groups=embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm2d(embed_dim),
            Rearrange('b c h w -> b (h w) c'),
            
        )
        self.conv_k, self.conv_v = [
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size_kv, stride_kv, padding_kv, groups=embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.BatchNorm2d(embed_dim),
                Rearrange('b c h w -> b (h w) c'),
            ) for _ in range(2)
        ]

        # Multi-head Attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, h, w):
        # Class tokens should not get involved in the convolutional projection
        x = self.layer_norm_1(x)
        if self.class_token:
            class_token, x = torch.split(x, [1, h * w], dim=1)

        # Rearrange x to (c, h, w) form for conv2d
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        q, k, v = self.conv_q(x), self.conv_k(x), self.conv_v(x)

        # Recover x to patch representation
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Put back the tokens
        if self.class_token:
            q, k, v = torch.cat((class_token, q), dim=1), torch.cat((class_token, k), dim=1), torch.cat((class_token, v), dim=1)
            x = torch.cat((class_token, x), dim=1)

        # Self-attention
        x = x + self.attention(q, k, v)[0]

        # MLP
        x = x + self.linear(self.layer_norm_2(x))
        return x


class CvTStage(nn.Module):
    def __init__(self, input_size, embed_dim, patch_size, patch_stride, patch_padding, hidden_dim, num_heads,
                 num_layers, class_token=False, dropout=0.1):
        super(CvTStage, self).__init__()

        self.input_size = input_size  # input_size (C, H, W)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_size[0] * self.patch_size ** 2  # channel x patch_h x patch_w

        # Convolutional Token Embedding
        self.conv_embedding = nn.Sequential(
            nn.Conv2d(input_size[0], embed_dim, patch_size, patch_stride, patch_padding),
            nn.GroupNorm(1, embed_dim),
        )

        # Add the classification token
        self.class_token = nn.Parameter(torch.randn(1, self.embed_dim)) if class_token else None

        # Transformer blocks x num_layers
        self.transformer = nn.ModuleList(
            [
                ConvTransformerBlock(embed_dim, hidden_dim, num_heads,
                                     3, 3, 1, 2, 1, 1,
                                     class_token, dropout
                                     ) for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # Convolutional Token Embedding
        x = self.conv_embedding(x)
        class_token = None

        # Get global b, c, h, w
        b, c, h, w = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')

        # Add class_token
        if self.class_token is not None:
            class_token = self.class_token.repeat(b, 1, 1)
            x = torch.cat([class_token, x], dim=1)

        for transformer in self.transformer:
            x = transformer(x, h, w)

        if self.class_token is not None:
            class_token, x = torch.split(x, [1, h * w], dim=1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x, class_token


class CvT(nn.Module):
    def __init__(self, config):
        super(CvT, self).__init__()
        # MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.Dropout(config['Classifier']['dropout']),
            nn.LayerNorm(config['Classifier']['embed_dim']),
            nn.Linear(config['Classifier']['embed_dim'], config['Classifier']['num_classes'])
        )

        # Three stages for extraction
        self.stage_1 = CvTStage(**config['Stage-1'])
        self.stage_2 = CvTStage(**config['Stage-2'])
        self.stage_3 = CvTStage(**config['Stage-3'])

    def forward(self, images):
        x, _ = self.stage_1(images)
        x, _ = self.stage_2(x)
        x, class_token = self.stage_3(x)

        y = self.mlp_head(class_token).squeeze(1)
        return y


if __name__ == "__main__":

    model = CvT(CvT_config)

    img = torch.arange(0, 3 * 224 * 224 * 10).reshape(10, 3, 224, 224).float()
    print("Shape of input:", img.shape)
    print("Shape of output:", model(img).shape)
    print('Trainable Parameters: %.3fM' % (sum(param.numel() for param in model.parameters()) / 1e6))

