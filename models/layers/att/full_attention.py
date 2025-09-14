
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from einops import rearrange, repeat
import torch.nn.functional as F


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.attention_dropout = attention_dropout
        # self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # scores.masked_fill_(attn_mask.mask, -np.inf)
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attn_mask.mask, neg_inf)
            
        A = F.dropout(torch.softmax(scale * scores, dim=-1), p=self.attention_dropout, training=self.training)
        V = torch.einsum("bhls, bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
        
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class CustomAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.dot_product = FullAttention(mask_flag=False, output_attention=False)
        self.att = AttentionLayer(self.dot_product, d_model=d_model, n_heads=n_heads)

    def forward(self, x, attn_mask=None):
        out, attn = self.att(x, x, x, attn_mask=attn_mask)
        return out

    
if __name__ == "__main__":
    # 超参数
    B, L, S, d_model, n_heads = 2, 4, 6, 16, 4

    # 构造输入
    x = torch.randn(B, L, d_model)   # [2, 4, 16]
    attn_mask = None   # 也可以用 TriangularCausalMask(B, L).mask

    att = CustomAttention(d_model=d_model, n_heads=n_heads)
    # 构造 AttentionLayer
    out, attn = att(x)

    print("out.shape:", out.shape)   # [B, L, d_model] -> [2, 4, 16]
    print("attn.shape:", attn.shape if attn is not None else None)  
    # [B, H, L, S] -> [2, 4, 4, 6]