import os


DEFAULT_DATASET = "psma"

PETCT_DATASETS = {
    "fdg": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG",
    "psma": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA",
}


def normalise_dataset_name(dataset):
    name = (dataset or DEFAULT_DATASET).lower()
    if name not in PETCT_DATASETS:
        choices = ", ".join(sorted(PETCT_DATASETS))
        raise ValueError(f"Unknown PET-CT dataset '{dataset}'. Choose one of: {choices}")
    return name


def resolve_petct_paths(dataset, train_data_path=None, test_data_path=None):
    dataset = normalise_dataset_name(dataset)
    root = PETCT_DATASETS[dataset]
    train_path = train_data_path or os.path.join(root, "train", "normal")
    test_path = test_data_path or os.path.join(root, "test")
    return train_path, test_path


def build_experiment_dir(save_root, dataset, modality):
    dataset = normalise_dataset_name(dataset)
    return os.path.join(save_root, f"{dataset}_{modality}")


def resolve_save_path(save_path, dataset, modality, use_dataset_subdir=True):
    if use_dataset_subdir:
        return build_experiment_dir(save_path, dataset, modality)
    return save_path


def checkpoint_name(epoch):
    return f"epoch_{int(epoch)}.pth"


def resolve_checkpoint_path(checkpoint_path, save_path, dataset, modality, epoch):
    if checkpoint_path:
        return checkpoint_path
    return os.path.join(build_experiment_dir(save_path, dataset, modality), checkpoint_name(epoch))


def select_heatmap_indices(total, count=0, save_all=False):
    if total <= 0:
        return []
    if save_all:
        return list(range(total))
    count = max(0, int(count or 0))
    return list(range(min(total, count)))
