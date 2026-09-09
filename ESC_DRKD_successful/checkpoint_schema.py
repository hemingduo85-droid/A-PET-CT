from collections.abc import Mapping


CHECKPOINT_FORMAT = "esc_drkd_complete"
CHECKPOINT_VERSION = 1


def build_complete_checkpoint(model, optimizer, **metadata):
    reserved = {"format", "format_version", "model", "optimizer"}
    collisions = reserved.intersection(metadata)
    if collisions:
        raise ValueError(f"Reserved checkpoint metadata keys: {sorted(collisions)}")
    return {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        **metadata,
    }


def require_complete_model_state(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Invalid ESC-DRKD checkpoint: expected a mapping.")
    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("format_version") != CHECKPOINT_VERSION:
        if "student" in checkpoint:
            raise RuntimeError(
                "Legacy student-only ESC-DRKD checkpoint cannot restore the training-time teacher; "
                "fix the checkpoint pipeline and retrain before evaluation."
            )
        raise RuntimeError(
            f"Unsupported ESC-DRKD checkpoint format. Expected {CHECKPOINT_FORMAT} "
            f"version {CHECKPOINT_VERSION}; retrain with the fixed training script."
        )

    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping):
        raise RuntimeError("Invalid ESC-DRKD checkpoint: complete model state is missing.")
    has_teacher = any(str(key).startswith("teacher.") for key in model_state)
    has_student = any(str(key).startswith("student.") for key in model_state)
    if not (has_teacher and has_student):
        raise RuntimeError(
            "Invalid ESC-DRKD checkpoint: complete model state is missing teacher or student parameters."
        )
    return model_state
