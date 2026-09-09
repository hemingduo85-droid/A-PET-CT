from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FlowCheckpointCompatTest(unittest.TestCase):
    def test_all_in_one_blocks_use_soft_permutation(self):
        source = (ROOT / "model" / "flow.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("permute_soft=True"), 2)


if __name__ == "__main__":
    unittest.main()
