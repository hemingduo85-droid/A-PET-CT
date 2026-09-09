from datetime import datetime

import pandas as pd
import os
import logging
import sys
import click
import torch
import warnings
import backbones
import glass
import utils
import petct_config
import metrics

"""
训练 PET-CT 伪 RGB（默认 PSMA，[CT, PET, PET]）：

bash shell/run-petct.sh

训练 FDG 单 PET：

DATASET=fdg MODALITY=pet bash shell/run-petct.sh

直接测试已有权重：

DATASET=psma MODALITY=petct bash shell/test-petct.sh

nohup env DATASET=fdg MODALITY=petct bash shell/test-petct.sh > test_fdg_ci.log 2>&1 &

95ci
cd /data/cyf/codes/A-PET-CT/GLASS-main
nohup env DATASET=psma MODALITY=petct bash shell/test-petct.sh > psma_npz.log 2>&1 &

nohup env DATASET=fdg MODALITY=petct bash shell/test-petct.sh > fdg_npz.log 2>&1 &

"""

@click.group(chain=True)
@click.option("--results_path", type=str, default="results", show_default=True,
              help="结果根目录，与 log_project/run_name 拼接；也可用 --save_dir 直接指定完整路径")
@click.option("--save_dir", type=str, default=None, show_default=True,
              help="权重与结果保存的完整目录（优先级最高），例如 checkpoints/psma_pet_ep30")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group", show_default=True,
              help="实验子目录名（与 log_project、run_name 一起组成保存路径）")
@click.option("--log_project", type=str, default="project", show_default=True,
              help="实验项目名，用于组成保存路径，例如 pet_psma")
@click.option("--run_name", type=str, default="test", show_default=True,
              help="本次运行名，用于组成保存路径，例如 pet_w50_30ep")
@click.option("--test", type=str, default="ckpt")
@click.option("--save_heatmaps", type=int, default=0, show_default=True,
              help="测试阶段保存热力图数量：0不保存，正数保存前N张，-1保存全部")
def main(**kwargs):
    pass


@main.command("net")
@click.option("--dsc_margin", type=float, default=0.5)
@click.option("--train_backbone", is_flag=True)
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--meta_epochs", type=int, default=30)
@click.option("--eval_epochs", type=int, default=30)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=1024)
@click.option("--pre_proj", type=int, default=1)
@click.option("--mining", type=int, default=1)
@click.option("--noise", type=float, default=0.015)
@click.option("--radius", type=float, default=0.75)
@click.option("--p", type=float, default=0.5)
@click.option("--lr", type=float, default=0.0001)
@click.option("--svd", type=int, default=0)
@click.option("--step", type=int, default=20)
@click.option("--limit", type=int, default=392)
def net(
        backbone_names,
        layers_to_extract_from,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize,
        meta_epochs,
        eval_epochs,
        dsc_layers,
        dsc_hidden,
        dsc_margin,
        train_backbone,
        pre_proj,
        mining,
        noise,
        radius,
        p,
        lr,
        svd,
        step,
        limit,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = []
        for idx in range(len(backbone_names)):
            layers_to_extract_from_coll.append(layers_to_extract_from)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_glass(input_shape, device):
        glasses = []
        for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            glass_inst = glass.GLASS(device)
            glass_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                meta_epochs=meta_epochs,
                eval_epochs=eval_epochs,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                train_backbone=train_backbone,
                pre_proj=pre_proj,
                mining=mining,
                noise=noise,
                radius=radius,
                p=p,
                lr=lr,
                svd=svd,
                step=step,
                limit=limit,
            )
            glasses.append(glass_inst.to(device))
        return glasses

    return "get_glass", get_glass


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", required=False, type=click.Path(file_okay=False))
@click.argument("aug_path", required=False, type=click.Path(file_okay=False))
@click.option("--petct_dataset", type=click.Choice(["fdg", "psma"], case_sensitive=False),
              default="psma", show_default=True,
              help="PET-CT 数据集 preset，会自动设置 FDG/PSMA 数据路径")
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=16, type=int, show_default=True)
@click.option("--resize", default=288, type=int, show_default=True)
@click.option("--imagesize", default=288, type=int, show_default=True)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--distribution", default=0, type=int)
@click.option("--mean", default=0.5, type=float)
@click.option("--std", default=0.1, type=float)
@click.option("--fg", default=1, type=int)
@click.option("--rand_aug", default=1, type=int)
@click.option("--downsampling", default=8, type=int)
@click.option("--augment", is_flag=True)
def dataset(
        name,
        data_path,
        aug_path,
        petct_dataset,
        subdatasets,
        batch_size,
        resize,
        imagesize,
        num_workers,
        rotate_degrees,
        translate,
        scale,
        brightness,
        contrast,
        saturation,
        gray,
        hflip,
        vflip,
        distribution,
        mean,
        std,
        fg,
        rand_aug,
        downsampling,
        augment,
):
    if name == "petct":
        petct_dataset = petct_config.normalize_dataset_key(petct_dataset)
        if not data_path:
            data_path = petct_config.resolve_dataset(petct_dataset)["data_path"]
    if not data_path:
        raise click.UsageError("data_path is required unless using dataset petct with --petct_dataset.")
    if not aug_path:
        aug_path = ""
    LOGGER.info("Dataset command: name=%s preset=%s data_path=%s aug_path=%s",
                name, petct_dataset, data_path, aug_path or "<none>")

    _DATASETS = {"petct": ["datasets.petct", "PETCTDataset"]}
    if name not in _DATASETS:
        raise click.UsageError("Only the PET-CT dataset is kept in this workspace. Use: dataset petct")
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed, test, get_name=name):
        dataloaders = []
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                aug_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            test_dataloader.name = get_name + "_" + subdataset
            test_dataloader.petct_dataset = petct_dataset if get_name == "petct" else None
            test_dataloader.modality = subdataset

            if test == 'ckpt':
                train_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    aug_path,
                    dataset_name=get_name,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.TRAIN,
                    seed=seed,
                    rotate_degrees=rotate_degrees,
                    translate=translate,
                    brightness_factor=brightness,
                    contrast_factor=contrast,
                    saturation_factor=saturation,
                    gray_p=gray,
                    h_flip_p=hflip,
                    v_flip_p=vflip,
                    scale=scale,
                    distribution=distribution,
                    mean=mean,
                    std=std,
                    fg=fg,
                    rand_aug=rand_aug,
                    downsampling=downsampling,
                    augment=augment,
                    batch_size=batch_size,
                )

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    prefetch_factor=2,
                    pin_memory=True,
                )

                train_dataloader.name = test_dataloader.name
                train_dataloader.petct_dataset = test_dataloader.petct_dataset
                train_dataloader.modality = test_dataloader.modality
                LOGGER.info(f"Dataset {subdataset.upper():^20}: train={len(train_dataset)} test={len(test_dataset)}")
            else:
                train_dataloader = test_dataloader
                LOGGER.info(f"Dataset {subdataset.upper():^20}: train={0} test={len(test_dataset)}")

            dataloader_dict = {
                "training": train_dataloader,
                "testing": test_dataloader,
            }
            dataloaders.append(dataloader_dict)

        print("\n")
        return dataloaders

    get_dataloaders.dataset_name = name
    get_dataloaders.petct_dataset = petct_dataset if name == "petct" else None
    get_dataloaders.modalities = list(subdatasets)
    return "get_dataloaders", get_dataloaders


@main.result_callback()
def run(
        methods,
        results_path,
        save_dir,
        gpu,
        seed,
        log_group,
        log_project,
        run_name,
        test,
        save_heatmaps,
):
    methods = {key: item for (key, item) in methods}
    save_heatmaps = petct_config.normalize_heatmap_count(save_heatmaps)

    if save_dir is None and getattr(methods["get_dataloaders"], "petct_dataset", None):
        modalities = getattr(methods["get_dataloaders"], "modalities", [])
        modality_label = "-".join(modalities) if len(modalities) > 1 else modalities[0]
        save_dir = petct_config.default_save_dir(
            results_path,
            methods["get_dataloaders"].petct_dataset,
            modality_label,
        )

    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite", save_dir=save_dir
    )
    LOGGER.info("Save directory (weights & results): %s", run_save_path)

    list_of_dataloaders = methods["get_dataloaders"](seed, test)

    device = utils.set_torch_device(gpu)

    result_collect = []
    data = {'Class': [], 'Distribution': [], 'Foreground': []}
    df = pd.DataFrame(data)
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        utils.fix_seeds(seed, device)
        dataset_name = dataloaders["training"].name
        imagesize = dataloaders["training"].dataset.imagesize
        glass_list = methods["get_glass"](imagesize, device)

        LOGGER.info(
            "Selecting dataset [{}] ({}/{}) {}".format(
                dataset_name,
                dataloader_count + 1,
                len(list_of_dataloaders),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        )

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, GLASS in enumerate(glass_list):
            flag = 0., 0., 0., 0., 0., -1.
            if GLASS.backbone.seed is not None:
                utils.fix_seeds(GLASS.backbone.seed, device)

            GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name)
            GLASS.set_heatmap_count(save_heatmaps)
            if test == 'ckpt':
                flag = GLASS.trainer(dataloaders["training"], dataloaders["testing"], dataset_name)
                if type(flag) == int:
                    row_dist = {'Class': dataloaders["training"].name, 'Distribution': flag, 'Foreground': flag}
                    df = pd.concat([df, pd.DataFrame(row_dist, index=[0])])

            if type(flag) != int:
                ev = GLASS.tester(dataloaders["testing"], dataset_name)
                row = {"dataset_name": dataset_name, **{k: v for k, v in ev.items() if k != "best_epoch"}}
                row["best_epoch"] = ev.get("best_epoch", -1)
                result_collect.append(row)

                if row["best_epoch"] > -1:
                    print(metrics.format_image_metrics(row), flush=True)
                    print(metrics.format_pixel_metrics(row), flush=True)
                    print(metrics.format_patient_metrics(row), flush=True)

                # save results csv after each category
                print("\n")
                result_metric_names = list(result_collect[-1].keys())[1:]
                result_dataset_names = [results["dataset_name"] for results in result_collect]
                result_scores = [list(results.values())[1:] for results in result_collect]
                utils.compute_and_store_final_results(
                    run_save_path,
                    result_scores,
                    result_metric_names,
                    row_names=result_dataset_names,
                )

    # save distribution judgment xlsx after all categories
    if len(df['Class']) != 0:
        os.makedirs('./datasets/excel', exist_ok=True)
        xlsx_path = './datasets/excel/' + dataset_name.split('_')[0] + '_distribution.xlsx'
        df.to_excel(xlsx_path, index=False)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
