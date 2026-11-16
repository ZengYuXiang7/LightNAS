"""Export one self-contained PyTorch source file per NAS-Bench-201 architecture."""

import argparse
import itertools
import json
import pickle
import random
from pathlib import Path
from string import Template

from tqdm import tqdm

from models.latency_data import OPERATION_NAMES, get_arch_str_from_arch_vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "f46486e21b71ae6459a700be720d7648b5429569"
EDGE_NAMES = ("0_1", "0_2", "1_2", "0_3", "1_3", "2_3")
OPERATOR_EXPRESSIONS = {
    "none": "Zero()",
    "skip_connect": "nn.Identity()",
    "nor_conv_1x1": "ReLUConvBN(channels, channels, 1, 1, 0)",
    "nor_conv_3x3": "ReLUConvBN(channels, channels, 3, 1, 1)",
    "avg_pool_3x3": "nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)",
}


def official_architectures():
    # 复现官方 gen_all 后 random.seed(88) 的编号；不采用本地字典顺序。
    architectures = []
    for operations in itertools.product(range(len(OPERATION_NAMES)), repeat=6):
        arch_str = get_arch_str_from_arch_vector(operations)
        architectures.append(arch_str)
    generator = random.Random(88)
    generator.shuffle(architectures)
    return architectures


def render_model(arch_index, arch_str, channels, cells_per_stage, num_classes):
    # 将边操作和依赖固化为源码；模型运行时不解析架构字符串。
    edge_definitions = []
    edge_index = 0
    for node in arch_str.split("+"):
        for edge in node.strip("|").split("|"):
            operation = edge.split("~")[0]
            expression = OPERATOR_EXPRESSIONS[operation]
            definition = f"        self.edge_{EDGE_NAMES[edge_index]} = {expression}"
            edge_definitions.append(definition)
            edge_index += 1

    # 显式展开完整网络；每个 cell 的参数独立而拓扑相同。
    network_layers = []
    previous_channels = channels
    for stage in range(3):
        stage_channels = channels * (2 ** stage)
        if stage > 0:
            layer = f"            ReductionBlock({previous_channels}, {stage_channels}),"
            network_layers.append(layer)
        for cell_index in range(cells_per_stage):
            layer = f"            Cell({stage_channels}),"
            network_layers.append(layer)
        previous_channels = stage_channels

    template_path = PROJECT_ROOT / "resources/nasbench201_model.py.template"
    template = Template(template_path.read_text(encoding="utf-8"))
    source = template.substitute(
        source_revision=SOURCE_REVISION,
        arch_index=arch_index,
        arch_str=arch_str,
        edge_definitions="\n".join(edge_definitions),
        channels=channels,
        network_layers="\n".join(network_layers),
        last_channels=previous_channels,
        num_classes=num_classes,
    )
    return source


def write_dataset_readme(output_dir, channels, cells_per_stage, num_classes):
    # 数据集说明同时交代输入方式、编号来源和参考实现的适用边界。
    text = f"""# NAS-Bench-201 standalone model sources

`models/model_00000.py` through `models/model_15624.py` are independent inputs.
Each file contains its own fixed cell, complete network and operator definitions.
Only PyTorch is required: instantiate `NASBench201()` from that file.
The forward result is `(pooled_features, logits)`.
Weights are newly initialized, not benchmark-trained weights.

Configuration: C={channels}, N={cells_per_stage} cells per stage,
num_classes={num_classes}. The standard CIFAR-10 input is [batch, 3, 32, 32].

`manifest.jsonl` links each source to `arch_index`, `arch_str`, `code_file` and
`graph_op_ids` from the project's existing data table. Indices reproduce the
official operation enumeration followed by a local random.Random(88) shuffle;
they are not the insertion order of nasbench201_complete.pickle.
No accuracy, latency, FLOPs or parameter-count labels are included in these inputs.

Reference: https://github.com/D-X-Y/AutoDL-Projects
Revision: {SOURCE_REVISION}
Files: xautodl/models/cell_infers/cells.py, cell_infers/tiny_network.py,
cell_operations.py, cell_searchs/genotypes.py; official indexing is also documented
by exps/NATS-Bench/tss-collect.py:traverse_net.
The MIT notice is retained in every model source.

The exporter preserves zero operations, identity edges, summation order,
ReLU-Conv-BN blocks and the full macro network. Graph branches describe data
dependencies, not a guarantee of parallel hardware execution. These are reference
PyTorch implementations, not reconstructed vendor-compiled binaries. Match the
runtime used to measure a latency label before making execution-level claims.

Exporter: utils/nasbench201_code.py in the LLGAT project.
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data_file", type=Path, default=PROJECT_ROOT / "data/nasbench201_complete.pickle")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "data/nasbench201_code")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--cells_per_stage", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--tqdm", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    # 只读取架构及图槽位编号，不将性能标签写入代码或输入索引。
    with args.data_file.open("rb") as data_file:
        architecture_table = pickle.load(data_file)
    architectures = official_architectures()
    args.output_dir.mkdir(parents=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir()
    manifest_path = args.output_dir / "manifest.jsonl"
    progress = tqdm(architectures, desc="Export NAS-Bench-201", disable=not args.tqdm)
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        for arch_index, arch_str in enumerate(progress):
            graph_op_ids = architecture_table[arch_str]["op_ids"]
            source = render_model(arch_index, arch_str, args.channels, args.cells_per_stage, args.num_classes)
            code_path = model_dir / f"model_{arch_index:05d}.py"
            code_path.write_text(source, encoding="utf-8")
            record = {
                "arch_index": arch_index,
                "arch_str": arch_str,
                "code_file": code_path.relative_to(args.output_dir).as_posix(),
                "graph_op_ids": graph_op_ids,
                "channels": args.channels,
                "cells_per_stage": args.cells_per_stage,
                "num_classes": args.num_classes,
                "reference_revision": SOURCE_REVISION,
            }
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed = arch_index + 1
            if not args.tqdm:
                if completed == 1 or completed % 1000 == 0:
                    print(f"Exported {completed}/{len(architectures)}", flush=True)
    write_dataset_readme(args.output_dir, args.channels, args.cells_per_stage, args.num_classes)
    print(f"Exported {len(architectures)}/{len(architectures)} model files", flush=True)
    display_dir = args.output_dir
    if args.output_dir.is_relative_to(PROJECT_ROOT):
        display_dir = args.output_dir.relative_to(PROJECT_ROOT)
    print(f"Output: {display_dir}", flush=True)


if __name__ == "__main__":
    main()
