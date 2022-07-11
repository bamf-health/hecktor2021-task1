import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning
import pytorch_lightning as pl
import SimpleITK as sitk
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from tqdm.auto import tqdm

from hecktor2021.hecktor_data import HecktorDataModule
from hecktor2021.hecktor_task import HecktorTask

USE_MULTIPROCESSING = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_path",
        help="paths to model checkpoint file",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save predicted labels to",
    )
    parser.add_argument(
        "--predict_aug",
        action="store_true",
        help="Use test time augmentation flips",
    )
    parser.add_argument(
        "--softmax",
        action="store_true",
        help="Save softmax values instead of binary mask",
    )

    parser = HecktorDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    config = parser.parse_args()
    return config


def save_pred(prediction: np.ndarray, ct_path: Path, save_dir: Path):
    # we need to switch axis order to sitk convention before saving with sitk
    prediction = np.transpose(prediction, (2, 1, 0))
    save_path = save_dir / f"{ct_path.name[:7]}.nii.gz"
    ct_img = sitk.ReadImage(str(ct_path))
    pred_img = sitk.GetImageFromArray(prediction)
    pred_img.CopyInformation(ct_img)
    sitk.WriteImage(pred_img, str(save_path))


if __name__ == "__main__":
    config = parse_args()
    config_vars = vars(config)
    hecktor_data = HecktorDataModule(**config_vars)
    trainer = pl.Trainer.from_argparse_args(config)

    assert Path(config.checkpoint_path).exists()
    checkpoint_path = config.checkpoint_path

    # single model prediction
    task = HecktorTask.load_from_checkpoint(
        checkpoint_path, predict_aug=config.predict_aug
    )
    prediction = trainer.predict(task, datamodule=hecktor_data)

    # pred_images = [p["pred"] for p in prediction]
    pred_images = prediction

    post_predict = AsDiscrete(argmax=True)
    np_prediction = []
    for p in pred_images:
        for dp in decollate_batch(p):
            if config.softmax:
                x = dp[1:, ...]  # drop background class
            else:
                x = post_predict(dp)
            np_prediction.append(x.squeeze().cpu().numpy())

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for p, meta, sample in tqdm(
        zip(np_prediction, prediction, hecktor_data.d_test),
        desc="Saving Predictions",
        total=len(np_prediction),
    ):
        save_pred(p, Path(sample["CT"]), config.output_dir)
