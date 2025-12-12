# coding : utf-8
# Author : Yuxiang Zeng
import torch
import torch.nn as nnx
from einops import rearrange


class Attention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.10):
        super().__init__()
        self.att = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )

    def forward(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        is_causal=False,
    ):
        # 如果你传的是 [B, H, N, M] 的 attn_mask
        if attn_mask is not None and attn_mask.dim() == 4:
            bs, h, n, m = attn_mask.shape
            attn_mask = attn_mask.reshape(bs * h, n, m)  # [B*H, N, M]

        out, weights = self.att(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,  # or None
            need_weights=need_weights,  # False -> weights=None
            attn_mask=attn_mask,  # or None
            is_causal=is_causal,
        )
        if need_weights:
            return out, weights
        else:
            return out


if __name__ == "__main__":
    inputs = torch.randn(1, 10, 64)
    model = Attention(d_model=64, num_heads=8, dropout=0.10)
    out = model(inputs)
    print(out.shape)

    inputs = torch.randn(2, 10, 64)  # [B=2, L=10, d_model=64]

    # 1. key_padding_mask 示例: 第二个样本后 3 个位置为 padding
    key_padding_mask = torch.tensor(
        [
            [False, False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, True, True, True],
        ]
    )

    # 2. attn_mask 示例（因果上三角 mask，全 batch 共用）
    L = inputs.size(1)
    attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)  # [L, L]

    # 3. 调用模型（需要注意，同时传入自定义 mask）
    model = Attention(d_model=64, num_heads=8, dropout=0.10)

    out, weights = model(
        inputs,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=False,
        average_attn_weights=False,  # 不做 head 平均 => 返回 [B, H, L, L]
        is_causal=False,  # 因果由 attn_mask 控制
    )

    print("output shape:", out.shape)
    print("weights shape:", weights.shape)
