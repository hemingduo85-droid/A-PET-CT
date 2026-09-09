import os


PETCT_DATASETS = {
    "fdg": {
        "data_path": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG",
    },
    "psma": {
        "data_path": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA",
    },
}


def normalize_dataset_key(dataset_key):
    key = dataset_key.lower()
    if key not in PETCT_DATASETS:
        available = ", ".join(sorted(PETCT_DATASETS))
        raise ValueError(f"Unknown PET-CT dataset '{dataset_key}'. Available: {available}")
    return key


def resolve_dataset(dataset_key):
    key = normalize_dataset_key(dataset_key)
    return {"name": key, **PETCT_DATASETS[key]}


def default_save_dir(root, dataset_key, modality):
    key = normalize_dataset_key(dataset_key)
    return os.path.join(root, f"{key}_{modality}")


def final_checkpoint_name(epoch):
    return f"ckpt_epoch_{int(epoch)}.pth"


def normalize_heatmap_count(count):
    count = int(count)
    if count < -1:
        raise ValueError("--save_heatmaps must be 0, a positive number, or -1 for all images")
    return count
