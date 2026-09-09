import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PETCT_SCRIPT = REPO_ROOT / "dinomaly2_petct.py"


class PetCTTrainProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = PETCT_SCRIPT.read_text(encoding="utf-8")
        cls.module = ast.parse(cls.source)
        cls.train_func = next(
            node
            for node in cls.module.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )

    def test_evaluates_only_after_training_loop(self):
        epoch_loops = [
            node
            for node in ast.walk(self.train_func)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "epoch"
        ]
        self.assertEqual(len(epoch_loops), 1)

        eval_names = {"evaluation_petct", "evaluation_petct_mulsen"}
        calls_inside_epoch_loop = [
            node
            for node in ast.walk(epoch_loops[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in eval_names
        ]

        self.assertEqual(calls_inside_epoch_loop, [])

    def test_saves_final_epoch_checkpoint_not_best_model(self):
        self.assertIn("'model.pth'", self.source)
        self.assertNotIn("'best_model.pth'", ast.get_source_segment(self.source, self.train_func))


if __name__ == "__main__":
    unittest.main()
