# LightNAS

Latency prediction on NAS-Bench-201, with LLGAT and baseline models.

## Data

The runtime expects `data/nasbench201_complete.pickle` and, for LLGAT,
`data/agu/devices.pickle`. These local data files are not included in Git.
See [data/README.md](data/README.md) and [data/DEVICE_TABLE.md](data/DEVICE_TABLE.md)
for the device labels and data protocol.

The evaluation set contains all valid architectures for the selected device,
including the training and validation sets. Reported metrics therefore describe
the complete architecture set rather than a held-out test set.

## RTX4090 environment

The RTX4090 environment uses Python 3.11, PyTorch 2.15.0 with CUDA 13.2,
and a locally built DGL package. The DGL wheel is stored in `.wheels/` on
RTX4090 and is not included in Git. See the [environment guide](docs/rtx4090-environment.md)
for its source and build options.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-rtx4090.txt
```

The 4090 baseline configuration is in
`run_exp/baselines/11.13/run-all-models.sh`; its latency labels are for a CPU
device, while the neural-network calculations use CUDA. The macOS one-epoch
smoke script is in `run_exp/smoke/11.14/baselines-epoch1-round1.sh`.
