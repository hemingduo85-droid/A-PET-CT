import unittest

from checkpoint_schema import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    build_complete_checkpoint,
    require_complete_model_state,
)


class _StateOwner:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class CheckpointSchemaTest(unittest.TestCase):
    def test_build_complete_checkpoint_stores_full_model_and_metadata(self):
        model_state = {
            "teacher.backbone.conv1.weight": "teacher-stem",
            "student.layer1.0.conv1.weight": "student-weight",
        }
        optimizer_state = {"state": {"step": 30}}

        checkpoint = build_complete_checkpoint(
            _StateOwner(model_state),
            _StateOwner(optimizer_state),
            epoch=30,
            tracer="PSMA",
            modalities=["ct", "pet"],
        )

        self.assertEqual(checkpoint["format"], CHECKPOINT_FORMAT)
        self.assertEqual(checkpoint["format_version"], CHECKPOINT_VERSION)
        self.assertIs(checkpoint["model"], model_state)
        self.assertIs(checkpoint["optimizer"], optimizer_state)
        self.assertEqual(checkpoint["epoch"], 30)
        self.assertEqual(checkpoint["tracer"], "PSMA")
        self.assertEqual(checkpoint["modalities"], ["ct", "pet"])

    def test_require_complete_model_state_accepts_versioned_checkpoint(self):
        model_state = {
            "teacher.backbone.conv1.weight": "teacher-stem",
            "student.layer1.0.conv1.weight": "student-weight",
        }
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "format_version": CHECKPOINT_VERSION,
            "model": model_state,
        }

        self.assertIs(require_complete_model_state(checkpoint), model_state)

    def test_require_complete_model_state_rejects_legacy_student_only_checkpoint(self):
        checkpoint = {"student": {"layer1.0.conv1.weight": "student-weight"}}

        with self.assertRaisesRegex(
            RuntimeError,
            "student-only.*cannot restore the training-time teacher.*retrain",
        ):
            require_complete_model_state(checkpoint)

    def test_require_complete_model_state_rejects_incomplete_model_state(self):
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "format_version": CHECKPOINT_VERSION,
            "model": {"student.layer1.0.conv1.weight": "student-weight"},
        }

        with self.assertRaisesRegex(RuntimeError, "missing teacher or student"):
            require_complete_model_state(checkpoint)


if __name__ == "__main__":
    unittest.main()
