import math

import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, head:int, dropout:float=0.2) -> None:
        super().__init__()

        assert d_model % head == 0, "d_model is must divisible by the head."

        self.d_model = d_model
        self.h = head
        self.dropout = nn.Dropout(dropout)

        self.d_k = d_model // head

        self.Wk = nn.Linear(d_model, d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x:torch.Tensor, mask=False) -> torch.Tensor:
        seq_len = x.shape[1]
        q = self.Wq(x) #(b, seq_len, d_model)
        k = self.Wk(x) #(b, seq_len, d_model)
        v = self.Wv(x) #(b, seq_len, d_model)

        # (b, h, seq_len, d_k)
        q = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)

        attention = q @ k.transpose(-1, -2) #(b, h, seq_len, seq_len)
        attention = attention / math.sqrt(self.d_k)
        if mask:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = mask.unsqueeze(0).unsqueeze(0) 
            attention.masked_fill_(mask == 0, -1e9)
        attention.softmax(dim=-1)
        attention = self.dropout(attention)
        x = attention @ v #(b, h, seq_len, d_k)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model) #(b, seq_len, d_model)
        x = self.Wo(x) #(b, seq_len, d_model)
        return x

if __name__ == "__main__":
    m = MultiHeadAttention(512, 8)
    i = torch.randn(1, 6, 512)
    out = m(i)
    print(out.shape)
















