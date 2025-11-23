from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


if __name__ == "__main__":
    head_size = 56
    rotary_dim = 56
    max_position_embeddings = 128
    base = 10000.0

    rope = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
    )

    # 假设序列长度为 10，batch_size = 2
    seq_len = 10
    batch_size = 2

    # positions: [L]，例如 0~9
    positions = torch.arange(seq_len, dtype=torch.long)  # [10]

    # query/key 形状： [L, B, D]
    query = torch.randn(seq_len, batch_size, head_size)
    key = torch.randn(seq_len, batch_size, head_size)

    print("input query shape :", query.shape)  # torch.Size([10, 2, 56])
    print("input key   shape :", key.shape)  # torch.Size([10, 2, 56])

    q_out, k_out = rope(positions, query, key)

    print("output query shape:", q_out.shape)  # 还是 [10, 2, 56]
    print("output key   shape:", k_out.shape)  # 还是 [10, 2, 56]

    # 看一下前几个数做参考
    print("input  query[0,0,:4]:", query[0, 0, :4])
    print("output query[0,0,:4]:", q_out[0, 0, :4])
