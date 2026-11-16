"""Check the exported corpus and compare models with pinned official sources.

Set NASBENCH201_REFERENCE_ROOT to the reference checkout's namespace-package root.
Run after exporting data/nasbench201_code on RTX4090.
"""

import ast
import json
import os
import pickle
import runpy
import sys
import unittest
from pathlib import Path

import torch

from utils.nasbench201_code import PROJECT_ROOT, official_architectures


REFERENCE_ROOT = Path(os.environ["NASBENCH201_REFERENCE_ROOT"])
sys.path.insert(0, str(REFERENCE_ROOT))

from xautodl.models.cell_infers.tiny_network import TinyNetwork
from xautodl.models.cell_searchs.genotypes import Structure


class NASBench201CodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 数值一致性属于 CPU 单元测试，不占用 GPU 或启动训练实验。
        torch.set_num_threads(1)
        cls.output_dir = PROJECT_ROOT / "data/nasbench201_code"
        cls.architectures = official_architectures()
        with (PROJECT_ROOT / "data/nasbench201_complete.pickle").open("rb") as data_file:
            cls.table = pickle.load(data_file)

    def test_official_architecture_indices(self):
        # 官方 traverse_net 给出了三个可独立核对的编号锚点。
        self.assertEqual(len(self.architectures), 15625)
        self.assertEqual(len(set(self.architectures)), 15625)
        self.assertEqual(set(self.architectures), set(self.table))
        self.assertEqual(
            self.architectures[0],
            "|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|",
        )
        self.assertEqual(
            self.architectures[9],
            "|avg_pool_3x3~0|+|none~0|none~1|+|skip_connect~0|none~1|nor_conv_3x3~2|",
        )
        self.assertEqual(
            self.architectures[123],
            "|avg_pool_3x3~0|+|avg_pool_3x3~0|nor_conv_1x1~1|+|none~0|avg_pool_3x3~1|nor_conv_3x3~2|",
        )

    def test_complete_export_and_standalone_syntax(self):
        # 逐文件核对架构、原图映射、语法和仅依赖 PyTorch 的输入边界。
        manifest_path = self.output_dir / "manifest.jsonl"
        with manifest_path.open(encoding="utf-8") as manifest_file:
            records = []
            for line in manifest_file:
                record = json.loads(line)
                records.append(record)
        self.assertEqual(len(records), 15625)
        self.assertEqual(len(list((self.output_dir / "models").glob("model_*.py"))), 15625)
        for arch_index, record in enumerate(records):
            arch_str = self.architectures[arch_index]
            self.assertEqual(record["arch_index"], arch_index)
            self.assertEqual(record["arch_str"], arch_str)
            self.assertEqual(record["graph_op_ids"], self.table[arch_str]["op_ids"])
            self.assertNotIn("latency", record)
            code_path = self.output_dir / record["code_file"]
            source = code_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(code_path))
            compile(tree, str(code_path), "exec")
            self.assertLessEqual(len(source.splitlines()), 250)
            self.assertNotIn("${", source)
            constants = {}
            imports = []
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                if isinstance(node, ast.Import):
                    for imported in node.names:
                        imports.append(imported.name)
                self.assertNotIsInstance(node, ast.ImportFrom)
            self.assertEqual(constants["ARCH_INDEX"], arch_index)
            self.assertEqual(constants["ARCHITECTURE"], arch_str)
            self.assertEqual(imports, ["torch", "torch.nn"])

    def test_reference_outputs_and_parameter_counts(self):
        # 覆盖五种纯操作架构、断路与混合分支，并核对完整网络输出。
        selected_indices = [0, 9, 42, 123, 1000, 15624]
        operation_names = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
        for operation in operation_names:
            arch_str = f"|{operation}~0|+|{operation}~0|{operation}~1|+|{operation}~0|{operation}~1|{operation}~2|"
            selected_indices.append(self.architectures.index(arch_str))
        generator = torch.Generator().manual_seed(7)
        inputs = torch.randn(2, 3, 32, 32, generator=generator)
        for arch_index in selected_indices:
            with self.subTest(arch_index=arch_index):
                arch_str = self.architectures[arch_index]
                code_path = self.output_dir / "models" / f"model_{arch_index:05d}.py"
                namespace = runpy.run_path(str(code_path))
                generated = namespace["NASBench201"]().eval()
                genotype = Structure.str2structure(arch_str)
                reference = TinyNetwork(16, 5, genotype, 10).eval()

                # 按官方注册顺序拷贝同一参数与 BN 缓冲区，隔离随机初始化。
                actual_state = list(generated.state_dict().values())
                reference_state = list(reference.state_dict().values())
                self.assertEqual(len(actual_state), len(reference_state))
                with torch.no_grad():
                    for tensor_index in range(len(actual_state)):
                        self.assertEqual(actual_state[tensor_index].shape, reference_state[tensor_index].shape)
                        actual_state[tensor_index].copy_(reference_state[tensor_index])
                    actual_features, actual_logits = generated(inputs)
                    expected_features, expected_logits = reference(inputs)
                torch.testing.assert_close(actual_features, expected_features, rtol=0, atol=0)
                torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
                self.assertEqual(actual_logits.shape, (2, 10))
                parameter_count = 0
                for parameter in generated.parameters():
                    parameter_count += parameter.numel()
                self.assertAlmostEqual(parameter_count / 1e6, self.table[arch_str]["params"], places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
