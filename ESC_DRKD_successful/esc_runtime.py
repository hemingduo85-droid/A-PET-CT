class FrozenTeacherTrainMixin:
    def train(self, mode=True):
        result = super().train(mode)
        self.teacher.eval()
        return result


def prepare_grayscale_resnet_state(state_dict):
    converted = state_dict.copy()
    if "conv1.weight" not in converted:
        raise KeyError("Pretrained ResNet state is missing conv1.weight")
    converted["conv1.weight"] = converted["conv1.weight"].mean(dim=1, keepdim=True)
    return converted
