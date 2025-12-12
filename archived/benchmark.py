import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=32, hidden=64, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def estimate_mlp_flops(model: nn.Module, input_dim: int):
    """
    粗略估算 MLP 的 FLOPs，只考虑 Linear 层：
    FLOPs ≈ 2 * in_dim * out_dim（乘加算两次）
    """
    flops = 0
    prev_dim = input_dim
    for m in model.modules():
        if isinstance(m, nn.Linear):
            out_dim = m.out_features
            flops += 2 * prev_dim * out_dim
            prev_dim = out_dim
    return flops


if __name__ == "__main__":
    mlp = SimpleMLP(in_dim=32, hidden=64, out_dim=10)

    # 1) benchmark 前向时间
    x = torch.rand(1, 32)
    t0 = benchmark.Timer(
        stmt="mlp(x)",
        setup="from __main__ import mlp",
        globals={"x": x},
    )
    print(t0.timeit(100))

    # 2) 参数量
    params = count_params(mlp)
    print(f"Total params: {params}")

    # 3) 粗略 FLOPs 估计（单次前向）
    flops = estimate_mlp_flops(mlp, input_dim=32)
    print(f"Estimated FLOPs per forward: {flops}")