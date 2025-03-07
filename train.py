import argparse
import random
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from datahandlers.dataloader import PETDataset, PETVOIDataset
from models.models import DLIFNet_MAX
from training.train import TrainUtilities
from utils.FileManager import FileManager


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=Path, nargs="+", help="Path to the config file(s).")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--aif", type=Path, required=True, help="Path to the AIF data.")
    data_group.add_argument("--img", type=Path, required=True, help="Path to the image data.")
    data_group.add_argument("--output", type=Path, required=True, help="Path to the output directory.")
    data_group.add_argument("--voi", type=Path, default=None, help="Path to the VOI data.")

    debug_group = parser.add_argument_group("debug")
    debug_group.add_argument("--dry-run", action="store_true", help="Do a dry run on one batch.")
    # debug_group.add_argument("--debug", action="store_true", help="Run in debug mode.")

    gpu_group = parser.add_argument_group("GPU")
    gpu_group.add_argument("--devices", type=int, default=1, help="Number of devices to use for optimization.")
    gpu_group.add_argument("--device", type=str, default="cuda", help="Device to use for optimization")

    wandb_group = parser.add_argument_group("wandb")
    wandb_group.add_argument("--group", type=str, default="test", help="Name for grouping in Wandb web UI.")

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)


def get_devices():
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(f"Device {i}: {torch.cuda.get_device_properties(i).name}")
    return devices


def load_config(config):
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


class PlotPredictionsCallback(Callback):
    def __init__(self, every_epoch=20):
        self.every_epoch = every_epoch

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log some predictions to wandb."""

        if batch_idx != 0:
            return

        if trainer.current_epoch % self.every_epoch != 0:
            return

        wandb_logger = trainer.logger.experiment

        time = batch["TIME"][0].cpu().detach().float().numpy()
        aif = batch["AIF"][0].cpu().detach().float().numpy()
        dlif = outputs["pred"][0].cpu().detach().float().numpy()

        fig, ax = plt.subplots()
        ax.set_title(f"Epoch {trainer.current_epoch} Loss {outputs['loss'].detach().cpu().numpy()}")
        ax.plot(time, aif, label="AIF")
        ax.plot(time, dlif, label="DLIF")
        ax.legend()
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("SUV [g/ml]")
        fig.tight_layout()

        wandb_logger.log({"val_prediction": fig})


def configure_callbacks(cfg, model_path):
    lr_callback = LearningRateMonitor(logging_interval="epoch")

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=model_path,
    #     save_top_k=1,
    #     monitor="val_loss",
    #     mode="min",
    #     filename="model_{epoch:04d}",
    # )

    plot_callback = PlotPredictionsCallback()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="model_{epoch:04d}",
        dirpath=model_path,
    )

    return [lr_callback, plot_callback, checkpoint_callback]


def configure_loggers(cfg, save_dir):
    wandb_logger = WandbLogger(
        name=f"Fold {cfg['fold']} run {cfg['run']}",
        project="Test DLIF",
        save_dir=save_dir,
        group=cfg["wandb_group"],
    )

    csv_logger = CSVLogger(save_dir=save_dir)

    return [wandb_logger, csv_logger]


def configure_dataset(cfg, args):

    if args.voi:
        train_dataset = PETVOIDataset(
            cfg,
            args.aif,
            args.img,
            args.voi,
            mode="train",
            sample_IDs=cfg["data"]["train_IDs"],
            numpy_dtype=np.float32,
        )
        val_dataset = PETVOIDataset(
            cfg,
            args.aif,
            args.img,
            args.voi,
            mode="val",
            sample_IDs=cfg["data"]["validation_IDs"],
            numpy_dtype=np.float32,
        )
    else:
        train_dataset = PETDataset(
            cfg,
            args.aif,
            args.img,
            mode="train",
            sample_IDs=cfg["data"]["train_IDs"],
        )
        val_dataset = PETDataset(
            cfg,
            args.aif,
            args.img,
            mode="val",
            sample_IDs=cfg["data"]["validation_IDs"],
        )
    train = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )
    val = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    return train, val


def train_run(config, args):
    cfg = load_config(config)

    model = cfg["model"]["name"]
    fold = cfg["fold"]
    run = cfg["run"]

    cfg["save_path"] = str(args.output)
    cfg["wandb_group"] = args.group
    cfg["device"] = args.device

    fm = FileManager(cfg)
    fm.update_fold(fold)
    fm.update_run(run)
    fold_path = fm.create_fold_folder()
    run_path = fm.create_run_folder()
    fm.save_config_file(savepath=run_path)
    model_path = fm.create_checkpoint_folder()

    utils = TrainUtilities(cfg)
    regression_criterion = utils.get_regression_criterion()
    optimizer = utils.get_optimizer()
    scheduler = utils.get_scheduler()

    model = DLIFNet_MAX(regression_criterion, optimizer, scheduler)

    train, val = configure_dataset(cfg, args)

    callbacks = configure_callbacks(cfg, model_path)
    loggers = configure_loggers(cfg, run_path)

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        precision=cfg["precision"],
        check_val_every_n_epoch=1,
        default_root_dir=cfg["save_path"],
        logger=loggers,
        callbacks=callbacks,
        accelerator=args.device,
        devices=args.devices,
        strategy="ddp",
        fast_dev_run=1 if args.dry_run else False,
    )

    trainer.fit(model, train, val)

    if not args.dry_run:
        wandb.finish()


if __name__ == "__main__":
    args = parse()

    for config in args.config:
        train_run(config, args)
