# most code from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        # dim_head: hidden dimension
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim) # head가 1개이면서 dim_head와 model input의 dimension이 같을 때 False

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        # x: (batch_size, n_tokens, dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv) # each (batch_size, heads, dim)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SelfiesTransformer(nn.Module):
    def __init__(self, vocab_dict, max_length, dim, n_classes, heads, mlp_dim, depth, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.dim = dim
        self.max_length = max_length

        n_token = len(vocab_dict)
        padding_idx = vocab_dict['[nop]']
        self.encode = nn.Embedding(n_token, dim, padding_idx=padding_idx)
        self.cls_token = nn.Parameter(torch.rand(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length + 1, dim)) # learnable
        self.dropout = nn.Dropout(emb_dropout)

        # self.to_latent = nn.Identity() # why?

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encode(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1) # (batch_size, max_length+1, dim)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 0] # final cls token only

        # x = self.to_latent(x)

        return self.mlp_head(x)
