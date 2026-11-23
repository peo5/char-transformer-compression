"""A minimal decoder-only transformer over characters, plus the data helpers that every script shares.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 \n.,;:!?'\"-()"
V = len(ALPHABET)
CONTEXT = 256
SIZES = {"0.1M": dict(d=64, layers=2, heads=4),       # parameter counts are printed by train.py
         "1M": dict(d=128, layers=5, heads=4),
         "10M": dict(d=320, layers=8, heads=8)}

BYTE_TO_ID = np.zeros(256, dtype=np.uint8)
BYTE_TO_ID[np.frombuffer(ALPHABET.encode(), dtype=np.uint8)] = np.arange(V, dtype=np.uint8)


def load_ids(path):
    """The text file as a read-only memory map of bytes; chunks() turns bytes into ids 0..V-1."""
    return np.memmap(path, dtype=np.uint8, mode="r")


def chunks(raw, starts):
    """Inputs and targets for chunks starting at `starts`: x = text[s : s+256], y = text[s+1 : s+257]."""
    block = np.stack([raw[s:s + CONTEXT + 1] for s in starts])
    block = torch.from_numpy(BYTE_TO_ID[block].astype(np.int64))
    return block[:, :-1], block[:, 1:]


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).transpose(1, 3).unbind(2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class CharTransformer(nn.Module):
    def __init__(self, d, layers, heads):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(CONTEXT, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, ids):
        """ids: (batch, t <= 256) -> logits (batch, t, V); position i sees ids[:, :i+1]."""
        x = self.tok(ids) + self.pos[:ids.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


def bits(logits, targets):
    """-log2 q(target) for every position, in float32: shape of targets."""
    return F.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none") / math.log(2)
