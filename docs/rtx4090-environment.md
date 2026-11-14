# RTX4090 environment

The project environment is `.venv` on RTX4090. It uses Python 3.11,
PyTorch 2.15.0 with CUDA 13.2, and DGL 2.5 built from
[DGL source commit `3d16000`](https://github.com/dmlc/dgl/tree/3d16000b4170fa741ed9e9667f22ba84d3493026).
The DGL wheel targets the RTX 4090's compute capability 8.9 and lives at
`.wheels/dgl-2.5-cp311-cp311-linux_x86_64.whl`; it is not committed to Git.

DGL's published compatibility matrix does not list PyTorch 2.15. On this
machine, the project data graph, batching, global-node construction,
`GraphConv`, `SAGEConv`, `GATConv`, and their gradients have been checked on
CUDA. This check is not a training run. GraphBolt, DGL Sparse, the PyTorch
tensor adapter, and LIBXSMM are disabled in this build because the current
project does not use them.

With the wheel already present, install the environment from the project root:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-rtx4090.txt
```

To rebuild the wheel on RTX4090, the CUDA 13.2 toolkit and a C/C++ compiler
must be available. The commands below build only the DGL components used by
this project. Run them from the project root:

```bash
PROJECT_DIR="$(pwd)"
uv venv --python 3.11 /tmp/lightnas-dgl-build-tools
uv pip install --python /tmp/lightnas-dgl-build-tools/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple cmake==3.30.9 ninja wheel setuptools cython==0.29.37
git clone https://github.com/dmlc/dgl.git /tmp/lightnas-dgl-source
git -C /tmp/lightnas-dgl-source checkout 3d16000b4170fa741ed9e9667f22ba84d3493026
git -C /tmp/lightnas-dgl-source submodule update --init third_party/GKlib third_party/METIS third_party/cccl third_party/dlpack third_party/dmlc-core third_party/nanoflann third_party/pcg third_party/tsl_robin_map
PATH=/tmp/lightnas-dgl-build-tools/bin:$PATH cmake -S /tmp/lightnas-dgl-source -B /tmp/lightnas-dgl-source/build -G Ninja -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.2 -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN=8.9 -DCUDA_ARCH_PTX= -DCMAKE_BUILD_TYPE=Release -DBUILD_TYPE=release -DBUILD_CPP_TEST=OFF -DBUILD_GRAPHBOLT=OFF -DBUILD_SPARSE=OFF -DBUILD_TORCH=OFF -DUSE_LIBXSMM=OFF
PATH=/tmp/lightnas-dgl-build-tools/bin:$PATH cmake --build /tmp/lightnas-dgl-source/build --parallel 8
mkdir -p "$PROJECT_DIR/.wheels"
cd /tmp/lightnas-dgl-source/python
/tmp/lightnas-dgl-build-tools/bin/python setup.py bdist_wheel --dist-dir "$PROJECT_DIR/.wheels"
cd "$PROJECT_DIR"
```
