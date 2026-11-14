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

Use Python 3.11 and the CUDA 12.1 packages listed in
`requirements-rtx4090.txt`:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-rtx4090.txt
```

If the DGL wheel host is unavailable, the RTX4090 project has a local wheel
in `.wheels/`, repacked from the [DGL team's CUDA 12.1 Conda package](https://conda.anaconda.org/dglteam/label/cuXX/linux-64/dgl-1.1.3.cu121-py311_0.tar.bz2).
For that machine, install the other requirements and then the local DGL wheel:

```bash
sed '/^dgl @ /d' requirements-rtx4090.txt > /tmp/lightnas-rtx4090-requirements.txt
uv pip install --python .venv/bin/python -r /tmp/lightnas-rtx4090-requirements.txt
uv pip install --python .venv/bin/python .wheels/dgl-1.1.3+cu121-cp311-cp311-linux_x86_64.whl
```

The 4090 baseline configuration is in
`run_exp/baselines/11.13/run-all-models.sh`; its latency labels are for a CPU
device, while the neural-network calculations use CUDA. The macOS one-epoch
smoke script is in `run_exp/smoke/11.14/baselines-epoch1-round1.sh`.
