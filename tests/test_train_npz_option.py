from pathlib import Path
import unittest


class TrainNpzOptionTest(unittest.TestCase):
    def test_npz_cache_is_guarded_by_explicit_flag(self):
        train_source = Path("train.py").read_text(encoding="utf-8-sig")

        self.assertIn('--save_npz', train_source)
        self.assertIn('if not args.save_npz:', train_source)
        self.assertIn('NPZ cache disabled', train_source)


if __name__ == "__main__":
    unittest.main()
