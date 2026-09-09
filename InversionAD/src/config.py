from copy import deepcopy


PETCT_DATASETS = {
    "fdg": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG",
    "psma": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA",
}


def resolve_petct_config(config):
    """Fill PETCT dataset-specific paths from config fields."""
    resolved = deepcopy(config)
    data = resolved.get("data", {})
    if data.get("dataset_name") != "petct":
        return resolved

    dataset_key = str(data.get("petct_dataset", data.get("category", "psma"))).lower()
    if dataset_key not in PETCT_DATASETS:
        valid = ", ".join(sorted(PETCT_DATASETS))
        raise ValueError(f"Invalid PETCT dataset '{dataset_key}'. Expected one of: {valid}")

    input_mode = str(data.get("input_mode", "dual")).lower()
    if input_mode not in ("ct", "pet", "dual"):
        raise ValueError("PETCT input_mode must be one of: ct, pet, dual")

    data["petct_dataset"] = dataset_key
    data["category"] = dataset_key
    data["data_root"] = data.get("data_root") or PETCT_DATASETS[dataset_key]
    if "data_roots" in data and dataset_key in data["data_roots"]:
        data["data_root"] = data["data_roots"][dataset_key]

    logging_cfg = resolved.setdefault("logging", {})
    save_root = logging_cfg.get("save_root")
    if save_root:
        logging_cfg["save_dir"] = f"{save_root.rstrip('/')}/{dataset_key}/{input_mode}"

    return resolved
