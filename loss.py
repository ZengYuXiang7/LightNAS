# train_kendall_reloss_tau_a.py
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# Utils: seed, device
# -----------------------------
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# ReLoss 网络 (标量回归)
# 输入: y_pred, y_true -> 输出: per-sample proxy loss
# 推荐输入拼接: [y_pred, y_true, diff, |diff|]
# -----------------------------
class RelossNet(nn.Module):
    def __init__(self, in_dim=4, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ELU()]
        layers += [nn.Linear(hidden, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred - y_true
        x = torch.cat([y_pred, y_true, diff, diff.abs()], dim=-1)
        return self.mlp(x)  # [bs,1]

# -----------------------------
# 可微 Kendall τ-a（连续回归用这个更简洁）
# -----------------------------
def soft_kendall_tau_a(L: torch.Tensor, T: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
    # L, T: [N,1]
    L = L.view(-1, 1)
    T = T.view(-1, 1)
    dL = L - L.t()                 # [N,N]
    dT = T - T.t()                 # [N,N]
    s  = dL * dT                   # 一致>0，不一致<0
    mask = torch.triu(torch.ones_like(s), diagonal=1).bool()
    s = s[mask]                    # 只取 i<j
    P = s.numel()
    if P == 0:
        return torch.zeros((), device=L.device)
    concord = torch.tanh(s / temp) # 平滑的 sign，范围 [-1,1]
    return concord.mean()          # 近似 τ-a

# -----------------------------
# 软 Spearman（仅用于日志参考）
# -----------------------------
def soft_rank(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    x = x.view(-1, 1)
    diff = x - x.t()
    P = torch.sigmoid(diff / tau)
    r = P.sum(dim=1, keepdim=True) + 1.0
    return r

def soft_spearman(a: torch.Tensor, b: torch.Tensor, tau: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    ra = soft_rank(a, tau)
    rb = soft_rank(b, tau)
    ra = (ra - ra.mean()) / (ra.std() + eps)
    rb = (rb - rb.mean()) / (rb.std() + eps)
    return (ra * rb).mean()

# -----------------------------
# 简单“主模型”用来产生 GM 样本（可替换为你自己的模型）
# -----------------------------
class ToyMainModel(nn.Module):
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 教师函数
def teacher_function(x: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.7, -1.1, 0.3, 0.9], device=x.device).view(1, -1)
    base = torch.sin(x @ w.t()) + 0.1 * x.norm(dim=1, keepdim=True)
    eps = 0.05 * torch.randn_like(base)
    return base + eps  # [bs,1]

# -----------------------------
# 数据迭代器：混合 GR & GM
# -----------------------------
class BatchIterator:
    def __init__(self, batch_size=256, p_random=0.5, device=None, main_model=None):
        self.bs = batch_size
        self.p_random = p_random
        self.device = device or get_device()
        self.main_model = main_model.to(self.device) if main_model is not None else None
        self.main_model.eval() if self.main_model is not None else None

    @torch.no_grad()
    def _make_gr_batch(self):
        y_true = torch.randn(self.bs, 1, device=self.device)
        noise  = 0.3 * torch.randn_like(y_true)
        y_pred = y_true + noise
        return y_pred, y_true

    @torch.no_grad()
    def _make_gm_batch(self):
        x = torch.randn(self.bs, 4, device=self.device)
        y_true = teacher_function(x)
        y_pred = self.main_model(x)
        y_pred = y_pred + 0.05 * torch.randn_like(y_pred)
        return y_pred, y_true

    def __call__(self):
        while True:
            use_gr = (random.random() < self.p_random) or (self.main_model is None)
            if use_gr:
                yield self._make_gr_batch()
            else:
                yield self._make_gm_batch()

# -----------------------------
# 训练 ReLoss：最大化 Kendall τ-a（最小化其相反数）+ 梯度惩罚
# metric: per-sample MAE；target = -metric (越大越好)
# -----------------------------
def train_reloss_kendall_tau_a(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[Info] device = {device}")

    # 1) 先把“主模型”简单拟合一下教师函数，生成更真实的 GM 样本
    main_model = ToyMainModel(in_dim=4, hidden=32).to(device)
    opt_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
    main_model.train()
    for _ in range(200):
        x = torch.randn(args.bs, 4, device=device)
        y = teacher_function(x)
        pred = main_model(x)
        loss = (pred - y).pow(2).mean()
        opt_main.zero_grad()
        loss.backward()
        opt_main.step()
    main_model.eval()

    # 2) 数据
    batch_iter = BatchIterator(batch_size=args.bs, p_random=args.p_random, device=device, main_model=main_model)
    data_gen = batch_iter()

    # 3) ReLoss 网络
    lossnet = RelossNet(in_dim=4, hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.Adam(lossnet.parameters(), lr=args.lr, weight_decay=args.wd)

    # 4) 训练
    lossnet.train()
    for step in range(1, args.steps + 1):
        y_pred, y_true = next(data_gen)                 # [bs,1]
        y_pred = y_pred.clone().detach().requires_grad_(True)

        L = lossnet(y_pred, y_true)                     # [bs,1]
        metric = (y_pred - y_true).abs()                # per-sample MAE
        target = -metric                                # 越大越好

        tau_a = soft_kendall_tau_a(L, target, temp=args.temp)
        loss_rank = -tau_a                              # 最大化 Kendall → 最小化其相反数

        # 梯度惩罚：||∂L/∂y_pred||_2 ≈ 1
        grad = torch.autograd.grad(L.sum(), y_pred, create_graph=True)[0]  # [bs,1]
        gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()

        loss = loss_rank + args.gp_lambda * gp
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_interval == 0:
            with torch.no_grad():
                spearman = soft_spearman(L, target, tau=args.spearman_tau)
            print(f"[{step:05d}] loss={loss.item():.4f}  -tau_a={loss_rank.item():.4f}  gp={gp.item():.4f}  tau_a={(-loss_rank).item():.4f}  spearman~{spearman.item():.4f}")

    # 5) 验证
    lossnet.eval()
    with torch.no_grad():
        y_pred_val, y_true_val = next(data_gen)
        L_val = lossnet(y_pred_val, y_true_val)
        metric_val = (y_pred_val - y_true_val).abs()
        target_val = -metric_val
        
        tau_a_val = soft_kendall_tau_a(L_val, target_val, temp=args.temp).item()
        spearman_val = soft_spearman(L_val, target_val, tau=args.spearman_tau).item()
        print(f"[Eval] Kendall tau_a={tau_a_val:.4f}  Spearman~{spearman_val:.4f}")

    # 6) 保存
    torch.save(lossnet.state_dict(), args.out)
    print(f"[Save] ReLoss (kendall τ-a) saved to: {args.out}")
    
    return lossnet

# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser("Train Kendall τ-a ReLoss for scalar regression")
    p.add_argument("--bs", type=int, default=256, help="batch size for training reloss")
    p.add_argument("--steps", type=int, default=4000, help="training steps for reloss")
    p.add_argument("--p_random", type=float, default=0.5, help="probability of sampling GR batch")
    p.add_argument("--hidden", type=int, default=64, help="hidden size of reloss net")
    p.add_argument("--depth", type=int, default=3, help="mlp depth (>=1)")
    p.add_argument("--lr", type=float, default=1e-2, help="learning rate for reloss")
    p.add_argument("--wd", type=float, default=1e-4, help="weight decay for reloss")
    p.add_argument("--gp_lambda", type=float, default=10.0, help="gradient penalty weight")
    p.add_argument("--temp", type=float, default=0.5, help="temperature for Kendall tanh")
    p.add_argument("--spearman_tau", type=float, default=1.0, help="tau for soft Spearman (logging)")
    p.add_argument("--log_interval", type=int, default=500, help="print interval")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--out", type=str, default="reloss_kendall_tau_a.pth")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    lossnet = train_reloss_kendall_tau_a(args)
    print(lossnet)