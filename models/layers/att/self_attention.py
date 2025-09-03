# coding : utf-8
# Author : Yuxiang Zeng
import torch
from einops import rearrange


class Attention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.10):
        super().__init__()
        self.att = torch.nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        out, weights = self.att(x, x, x, key_padding_mask=key_padding_mask)
        return out

if __name__ == '__main__':
    inputs = torch.randn(1, 10, 64)
    model = Attention(d_model = 64, num_heads = 8, dropout = 0.10)
    out = model(inputs)
    print(out.shape)