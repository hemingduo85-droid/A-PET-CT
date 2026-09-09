DATASET_CONFIGS = {
    "psma": {
        "name": "psma",
        "data_root": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA",
        "checkpoint_prefix": "rd4ad_autopet_psma",
    },
    "fdg": {
        "name": "fdg",
        "data_root": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG",
        "checkpoint_prefix": "rd4ad_autopet_fdg",
    },
}


def get_dataset_config(dataset):
    key = dataset.lower()
    if key not in DATASET_CONFIGS:
        valid = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset '{dataset}'. Valid choices: {valid}")
    return DATASET_CONFIGS[key]


def parse_modalities(modality, modalities):
    if modalities.strip():
        values = [m.strip().lower() for m in modalities.split(",") if m.strip()]
    else:
        values = [modality.lower()]
    invalid = [m for m in values if m not in {"pet", "ct"}]
    if invalid:
        raise ValueError(f"Invalid modalities: {invalid}. Use pet, ct, or pet,ct.")
    return values


def resolve_input_mode(modalities, input_mode):
    if input_mode != "auto":
        return input_mode
    if set(modalities) == {"pet", "ct"} and len(modalities) == 2:
        return "ct_pet_pet"
    return "rgb"


def checkpoint_tag(modalities, input_mode):
    mode = resolve_input_mode(modalities, input_mode)
    if mode == "ct_pet_pet":
        return "ct_pet_pet"
    if mode == "concat":
        return "concat_" + "_".join(modalities)
    return "_".join(modalities)


def checkpoint_path(dataset_config, modalities, input_mode, checkpoint_dir="./checkpoints"):
    tag = checkpoint_tag(modalities, input_mode)
    return f"{checkpoint_dir}/{dataset_config['checkpoint_prefix']}_{tag}.pth"


def input_channels(modalities, input_mode):
    mode = resolve_input_mode(modalities, input_mode)
    if mode in {"rgb", "ct_pet_pet"}:
        return 3
    if mode == "concat":
        return 3 * len(modalities)
    raise ValueError(f"Unsupported input mode: {input_mode}")
