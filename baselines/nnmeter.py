"""
Layer-wise latency profiler (no-recursion hooks) + simple LUT estimator.

- Records per-layer latency using forward_pre_hook / forward_hook
- Works on CPU and CUDA (uses cuda Events when available)
- Exports per-layer CSV and a summary TXT
- Includes a tiny demo model; swap with your own nn.Module
"""

from __future__ import annotations
import csv
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import List, Tuple, Dict

import torch
import torch.nn as nn


# =========================
# Example model (replaceable)
# =========================
class TinyNet(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dw = nn.Sequential(  # depthwise + pointwise
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.dw(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


# =========================
# Layer feature schema
# =========================
@dataclass
class LayerFeat:
    name: str
    type: str
    # common
    in_hw: int | None = None
    cin: int | None = None
    cout: int | None = None
    # conv
    k: int | None = None
    s: int | None = None
    groups: int | None = None
    # linear
    in_features: int | None = None
    out_features: int | None = None
    # pool
    pool_k: int | None = None
    pool_s: int | None = None
    # result
    latency_ms: float | None = None

    def signature(self) -> Tuple:
        t = self.type
        if t == "Conv2d":
            return (t, self.in_hw, self.cin, self.cout, self.k, self.s, self.groups)
        if t == "Linear":
            return (t, self.in_features, self.out_features)
        if t in {"BatchNorm2d", "ReLU", "SiLU", "LeakyReLU"}:
            return (t, self.in_hw, self.cin)
        if t in {"MaxPool2d", "AvgPool2d"}:
            return (t, self.in_hw, self.pool_k, self.pool_s)
        return (t, self.in_hw, self.cin, self.cout)


# =========================
# Layer-wise profiler (no recursion)
# =========================
class LayerWiseProfiler:
    def __init__(self, model: nn.Module, device: torch.device, warmup: int = 10, iters: int = 50):
        self.m = model.to(device).eval()
        self.device = device
        self.warmup = max(0, warmup)
        self.iters = max(1, iters)

        self._start: Dict[int, torch.cuda.Event | float] = {}
        self._acc_ms: Dict[str, float] = defaultdict(float)
        self._feats: Dict[str, LayerFeat] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _is_leaf(mod: nn.Module) -> bool:
        return len(list(mod.children())) == 0

    def _mk_feat(self, name: str, mod: nn.Module, x, y) -> LayerFeat:
        ft = self._feats.get(name) or LayerFeat(name=name, type=mod.__class__.__name__)
        self._feats[name] = ft

        # Shapes
        y0 = y[0] if isinstance(y, tuple) else y
        if isinstance(y0, torch.Tensor):
            if y0.dim() == 4:
                _, c, h, w = y0.shape
                ft.in_hw = h
                ft.cout = c
            elif y0.dim() == 2:
                _, f = y0.shape
                ft.out_features = f

        x0 = x[0] if isinstance(x, tuple) else x
        if isinstance(x0, torch.Tensor):
            if x0.dim() == 4:
                _, cin, h, w = x0.shape
                ft.cin = cin
                ft.in_hw = h
            elif x0.dim() == 2:
                _, f = x0.shape
                ft.in_features = f

        # Params
        if isinstance(mod, nn.Conv2d):
            ft.k = mod.kernel_size[0]
            ft.s = mod.stride[0]
            ft.groups = mod.groups
            ft.cin = mod.in_channels
            ft.cout = mod.out_channels
        elif isinstance(mod, nn.Linear):
            ft.in_features = mod.in_features
            ft.out_features = mod.out_features
        elif isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d)):
            k = mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0]
            s = mod.stride if isinstance(mod.stride, int) else (mod.stride or k)
            ft.pool_k = k
            ft.pool_s = s

        return ft

    def setup(self):
        if self._handles:
            return
        use_cuda = self.device.type == "cuda"

        for name, mod in self.m.named_modules():
            if not name or not self._is_leaf(mod):
                continue

            if use_cuda:
                def pre_hook(m, inp, _name=name):
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    self._start[id(m)] = start

                def post_hook(m, inp, out, _name=name):
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    torch.cuda.synchronize(self.device)
                    elapsed = self._start[id(m)].elapsed_time(end)  # ms
                    self._acc_ms[_name] += float(elapsed)
                    self._mk_feat(_name, m, inp, out)
            else:
                def pre_hook(m, inp, _name=name):
                    self._start[id(m)] = time.perf_counter()

                def post_hook(m, inp, out, _name=name):
                    end = time.perf_counter()
                    elapsed = (end - self._start[id(m)]) * 1000.0
                    self._acc_ms[_name] += float(elapsed)
                    self._mk_feat(_name, m, inp, out)

            h1 = mod.register_forward_pre_hook(pre_hook)
            h2 = mod.register_forward_hook(post_hook)
            self._handles.extend([h1, h2])

    def teardown(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._start.clear()

    def run(self, x: torch.Tensor) -> List[LayerFeat]:
        x = x.to(self.device)

        with torch.no_grad():
            # warmup
            for _ in range(self.warmup):
                _ = self.m(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            # measure
            for _ in range(self.iters):
                _ = self.m(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        feats: List[LayerFeat] = []
        for name, ft in self._feats.items():
            ft.latency_ms = self._acc_ms[name] / self.iters
            feats.append(ft)
        return feats


# =========================
# Simple LUT estimator
# =========================
class LayerLUT:
    def __init__(self):
        self.table: Dict[Tuple, float] = {}

    def fit(self, feats: List[LayerFeat]):
        for f in feats:
            if f.latency_ms is None:
                continue
            sig = f.signature()
            if sig not in self.table:
                self.table[sig] = f.latency_ms
            else:
                self.table[sig] = 0.5 * (self.table[sig] + f.latency_ms)

    def predict_layer_ms(self, feat: LayerFeat) -> float:
        sig = feat.signature()
        if sig in self.table:
            return self.table[sig]
        # fallback: type-wise average
        vals = [v for k, v in self.table.items() if k and k[0] == feat.type]
        if vals:
            return sum(vals) / len(vals)
        return 0.0

    def estimate_model_ms(self, feats: List[LayerFeat]) -> float:
        return sum(self.predict_layer_ms(f) for f in feats)


# =========================
# Utilities
# =========================
def export_csv(feats: List[LayerFeat], path: str = "./per_layer_latency.csv"):
    if not feats:
        return
    keys = list(asdict(feats[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for ft in feats:
            w.writerow(asdict(ft))


def measure_model_latency_ms(model: nn.Module, x: torch.Tensor, device: torch.device, warmup: int = 10, iters: int = 50) -> float:
    model = model.to(device).eval()
    x = x.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.time() - t0) * 1000.0 / iters


# =========================
# Demo main
# =========================
def main():
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyNet()
    x = torch.randn(1, 3, 224, 224)

    # real whole-model latency
    real_ms = measure_model_latency_ms(model, x, device, warmup=10, iters=50)

    # layer-wise measure (no recursion)
    profiler = LayerWiseProfiler(model, device, warmup=10, iters=50)
    profiler.setup()
    feats = profiler.run(x)
    profiler.teardown()

    # LUT baseline
    lut = LayerLUT()
    lut.fit(feats)
    est_ms = lut.estimate_model_ms(feats)

    # export
    export_csv(feats, "per_layer_latency.csv")
    with open("./model_latency_report.txt", "w") as f:
        f.write(f"Device: {device.type}\n")
        f.write(f"Real total latency (ms): {real_ms:.3f}\n")
        f.write(f"Layer-wise estimated total latency (ms): {est_ms:.3f}\n")
        f.write("Note: layer-wise sum ignores fusion/overlap; expect bias vs. real.\n")

    print("=== Layer-wise Latency Profiling ===")
    print(f"Device: {device.type}")
    print(f"Real total latency (ms): {real_ms:.3f}")
    print(f"Layer-wise estimated total latency (ms): {est_ms:.3f}")
    print("Per-layer CSV -> per_layer_latency.csv")
    print("Report       -> model_latency_report.txt")


if __name__ == "__main__":
    main()