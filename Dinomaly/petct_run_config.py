import os


def resolve_checkpoint_path(args):
    if getattr(args, "ckpt_path", None):
        return args.ckpt_path
    return os.path.join(args.save_dir, args.save_name, f"epoch_{args.num_epochs:02d}_model.pth")


def resolve_anomaly_dir(args):
    if getattr(args, "anomaly_dir", None):
        return args.anomaly_dir
    return os.path.join(args.save_dir, args.save_name, "anomaly_maps")
