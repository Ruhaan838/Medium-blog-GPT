from models.attention import MultiHeadAttention

import torch
from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.2):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x:torch.Tensor):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self, d_model:int, d_ff:int, head_size):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.maskmulti = MultiHeadAttention(d_model, head_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.multi = MultiHeadAttention(d_model, head_size)
        self.norm3 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, d_ff)

    def forward(self, x:torch.Tensor):

        res = x
        x = self.norm1(x)
        x = self.maskmulti(x, True)
        x += res

        res = x
        x = self.norm2(x)
        x = self.multi(x)
        x += res

        res = x
        x = self.norm3(x)
        x = self.feedforward(x)
        x += res

        return  x

class GPT2(nn.Module):
    def __init__(self, N:int, vocab_size:int, d_model, d_ff, head_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Decoder(d_model, d_ff, head_size) for _ in range(N)])
        self.proj = nn.Linear(d_model, vocab_size)

        self.apply(self._init_parameter)

    def _init_parameter(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def decode(self, x):
        txt = self.embedding(x)
        pos = self.positional_embedding(torch.arange(self.seq_len, device=x.device))
        x = txt + pos
        for layer in self.blocks:
            x = layer(x)
        return  x

    def projection(self, x):
        x = self.proj(x)
        return x

if __name__ == "__main__":
    a = torch.randint(0, 6, (1, 12))
    m = GPT2(2, 6, 512, 2048, 8, 12)
    out = m.decode(a)
    print(out.shape)
    out = m.projection(out)
    print(out.shape)
