import unittest

from esc_runtime import FrozenTeacherTrainMixin, prepare_grayscale_resnet_state


class _FakeWeight:
    def __init__(self):
        self.mean_calls = []

    def mean(self, *args, **kwargs):
        self.mean_calls.append((args, kwargs))
        return "gray-conv1"


class _FakeTeacher:
    def __init__(self):
        self.training = True
        self.eval_calls = 0

    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self


class _BaseModel:
    def __init__(self):
        self.training = False
        self.teacher = _FakeTeacher()

    def train(self, mode=True):
        self.training = bool(mode)
        self.teacher.training = bool(mode)
        return self


class _TeacherStudentModel(FrozenTeacherTrainMixin, _BaseModel):
    pass


class EscRuntimeTest(unittest.TestCase):
    def test_prepare_grayscale_resnet_state_uses_rgb_channel_mean(self):
        rgb_weight = _FakeWeight()
        original = {"conv1.weight": rgb_weight, "layer1.weight": "feature-weight"}

        converted = prepare_grayscale_resnet_state(original)

        self.assertIsNot(converted, original)
        self.assertEqual(converted["conv1.weight"], "gray-conv1")
        self.assertEqual(converted["layer1.weight"], "feature-weight")
        self.assertIs(original["conv1.weight"], rgb_weight)
        self.assertEqual(
            rgb_weight.mean_calls,
            [((), {"dim": 1, "keepdim": True})],
        )

    def test_model_train_keeps_teacher_in_eval_mode(self):
        model = _TeacherStudentModel()

        returned = model.train(True)

        self.assertIs(returned, model)
        self.assertTrue(model.training)
        self.assertFalse(model.teacher.training)
        self.assertEqual(model.teacher.eval_calls, 1)


if __name__ == "__main__":
    unittest.main()
