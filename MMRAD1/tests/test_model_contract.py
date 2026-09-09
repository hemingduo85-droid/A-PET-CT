import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ModelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "model.py").read_text(encoding="utf-8-sig")
        cls.tree = ast.parse(cls.source)

    def test_attention_uses_batch_first(self):
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "MultiheadAttention"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertIn("batch_first", keywords)
        self.assertIsInstance(keywords["batch_first"], ast.Constant)
        self.assertTrue(keywords["batch_first"].value)

    def test_patch_embedding_converts_nchw_to_bhwc(self):
        self.assertIn("x = self.patch_embed(x).permute(0, 2, 3, 1).contiguous()", self.source)

    def test_drop_path_is_disabled_during_evaluation_and_rescaled_during_training(self):
        self.assertIn("if self.drop_prob == 0. or not self.training:", self.source)
        self.assertIn("/ keep_prob", self.source)

    def test_layer_norm_uses_learned_affine_parameters(self):
        self.assertIn("self.gamma[:, None, None] * x", self.source)
        self.assertIn("self.beta[:, None, None]", self.source)


if __name__ == "__main__":
    unittest.main()
