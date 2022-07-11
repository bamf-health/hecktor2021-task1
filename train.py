import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
from pathlib import Path

import monai.config
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import hecktor2021
from hecktor2021.hecktor_data import HecktorDataModule
from hecktor2021.hecktor_task import HecktorTask

torch.backends.cudnn.benchmark = True


def resume_experiment(config: argparse.Namespace):
    print("resuming training")
    task = None
    task = HecktorTask.load_from_checkpoint(config.resume_from_checkpoint)
    if task is None:
        print(f"Unable to resume Task from checkpoint {config.resume_from_checkpoint}")
        sys.exit(-1)
    restore_experiement_paths(config, task)

    return task


def main(config: argparse.Namespace):
    task = None
    if config.resume_from_checkpoint:
        task = resume_experiment(config)

    if task is None:
        setup_experiement_paths(config)

    config_vars = vars(config)

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_path,
        filename=config.exp_name,
        save_last=True,
        save_top_k=1,
        monitor="val/dice",
        mode="max",
    )
    # early_stopping_callback = EarlyStopping(monitor="val/dice", patience=50, mode="max")

    # ------------
    # set up data and task
    # ------------
    hecktor_data = HecktorDataModule(**config_vars)

    if task is None:
        task = HecktorTask(**config_vars)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(
        config,
        callbacks=[
            checkpoint_callback,
            # early_stopping_callback,
        ],
        num_sanity_val_steps=1,
        precision=16,
        # accelerator="ddp",
        # plugins="ddp_sharded",
        # auto_lr_find=True,
        # auto_scale_batch_size="binsearch",
    )

    # find learning rate and batch size, if requested in trainer args
    old_lr = task.hparams.lr
    old_bs = task.hparams.batch_size
    trainer.tune(task, datamodule=hecktor_data)  # find a good learning rate
    if old_lr != task.hparams.lr:
        print(f"Learning rate changed from {old_lr} to {task.hparams.lr}")
    if old_bs != task.hparams.batch_size:
        print(f"Batch size changed from {old_bs} to {task.hparams.batch_size}")

    trainer.fit(task, datamodule=hecktor_data)

    # ------------
    # testing
    # ------------
    # trainer.test(task, datamodule=hecktor_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = HecktorDataModule.add_model_specific_args(parser)
    parser = HecktorTask.add_model_specific_args(parser)

    # miscellaneous
    misc_parser = parser.add_argument_group("Miscellaneous")
    misc_parser.add_argument("--seed", type=int, default=1805473289)
    misc_parser.add_argument("--verbose", type=int, default=0)
    misc_parser.add_argument("--experiment_suffix")

    parser = pl.Trainer.add_argparse_args(parser)

    config = parser.parse_args()

    # set experiment name
    config.exp_name = config.model_type
    if config.experiment_suffix:
        config.exp_name += "_" + config.experiment_suffix

    return config


def restore_experiement_paths(config: argparse.Namespace, task: HecktorTask):
    """setup other values based on command args"""
    # Setup Directories
    config.exp_path = task.hparams["exp_path"]
    config.model_path = task.hparams["model_path"]
    config.predicitons = task.hparams["predicitons"]
    config.predicitons_original_resolution = task.hparams[
        "predicitons_original_resolution"
    ]


def setup_experiement_paths(config: argparse.Namespace, experiment_id=""):
    """setup other values based on command args"""
    # Setup Directories
    exp_dir_name = (
        f"{experiment_id}_{config.exp_name}" if experiment_id else config.exp_name
    )
    config.exp_path = (
        Path(hecktor2021.__file__).parent.parent / "experiments" / exp_dir_name
    )
    config.model_path = config.exp_path / "models"
    config.predicitons = config.exp_path / "predicitons"
    config.predicitons_original_resolution = config.exp_path / "for_submission"

    # Create directories
    config.model_path.mkdir(exist_ok=True, parents=True)
    config.predicitons.mkdir(exist_ok=True, parents=True)
    config.predicitons_original_resolution.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    config = parse_args()
    pl.seed_everything(config.seed)
    monai.config.print_config()
    main(config)
