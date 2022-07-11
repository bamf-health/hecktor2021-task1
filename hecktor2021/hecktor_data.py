from math import radians
from pathlib import Path
from typing import List, Optional, Sequence

import pytorch_lightning
import torch
from monai.data import CacheDataset, Dataset, list_data_collate, load_decathlon_datalist
from monai.transforms import (
    AddChanneld,
    Compose,
    ConcatItemsd,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    ScaleIntensityRanged,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.transforms.transform import MapTransform


class HecktorDataModule(pytorch_lightning.LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("data")
        parser.add_argument(
            "--datasplit_json",
            type=Path,
            help="Dataset split to train on",
        )
        parser.add_argument(
            "--datasplit_base_dir",
            type=str,
            help="paths in datasplit_json are relative to this dir. Defaults to dir that datasplit_json is in",
        )

        parser.add_argument(
            "--patch_size",
            type=int,
            nargs=3,
            default=(128, 128, 128),
            help="patch size of images used for training",
        )

        parser.add_argument(
            "--spacing_size",
            type=float,
            nargs=3,
            default=(1.0, 1.0, 1.0),
            help="resample images to this spacing",
        )

        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--num_workers", type=int, default=16)
        return parent_parser

    def __init__(
        self,
        datasplit_json: Path,
        *,
        datasplit_base_dir: str = None,
        patch_size: Sequence[int] = (128, 128, 128),
        spacing_size: Sequence[float] = (1.0, 1.0, 1.0),
        batch_size: int = 2,
        num_workers: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ct_bounds = (-250, 250)
        self.patch_size = patch_size
        self.spacing_size = spacing_size

        self.d_train = load_decathlon_datalist(
            str(datasplit_json),
            True,
            "training",
            base_dir=datasplit_base_dir,
        )
        self.d_test = load_decathlon_datalist(
            str(datasplit_json),
            True,
            "test",
            base_dir=datasplit_base_dir,
        )
        self.d_val = load_decathlon_datalist(
            str(datasplit_json),
            True,
            "validation",
            base_dir=datasplit_base_dir,
        )

        print((100 * "*"), "\n")
        print(f"Training Subjects {len(self.d_train)}")
        print(f"Validation Subjects {len(self.d_val)}")
        print(f"Inference Subjects {len(self.d_test)}")
        print((100 * "*"), "\n")

    def setup_train_val(self):
        rotate_range = (radians(-30), radians(30))

        load_transforms: List[MapTransform] = [
            LoadImaged(keys=["CT", "PT", "label"]),
            AddChanneld(keys=["CT", "PT", "label"]),
            Orientationd(keys=["CT", "PT", "label"], axcodes="LPS"),
            Spacingd(
                keys=["CT", "PT", "label"],
                pixdim=self.spacing_size,
                mode=("bilinear", "bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["CT"],
                a_min=self.ct_bounds[0],
                a_max=self.ct_bounds[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["CT", "PT", "label"], source_key="CT"),
            NormalizeIntensityd(keys=["CT", "PT"]),
            SpatialPadd(keys=["CT", "PT"], spatial_size=self.patch_size, mode="edge"),
            ConcatItemsd(keys=["CT", "PT"], name="image"),
            SelectItemsd(keys=["image", "label"]),
        ]

        final_transforms: List[MapTransform] = [
            SpatialPadd(
                keys=["image", "label"], spatial_size=self.patch_size
            ),  # pad if the image is smaller than patch
            ToTensord(keys=["image", "label"]),
        ]

        rand_transforms: List[MapTransform] = [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.patch_size,
                pos=1,
                neg=1,
                num_samples=4,
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.9,
                max_zoom=1.2,
                mode=("trilinear", "nearest"),
                align_corners=(True, None),
                prob=0.15,
            ),
            RandAffined(
                keys=["image", "label"],
                prob=0.15,
                rotate_range=[rotate_range, rotate_range, rotate_range],
                scale_range=[0.7, 1.4],
                mode=("bilinear", "nearest"),
                cache_grid=True,
                spatial_size=self.patch_size,
                # as_tensor_output=False,
            ),
            Rand3DElasticd(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.1,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="border",
                as_tensor_output=False,
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=0.15,
                mean=0.0,
                std=0.1,
            ),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                approx="erf",
                prob=0.1,
            ),
            RandScaleIntensityd(keys=["image"], factors=(0.7, 1.3), prob=0.15),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
        ]

        train_transforms = Compose(load_transforms + rand_transforms + final_transforms)
        val_transforms = Compose(load_transforms + final_transforms)

        self.train_ds = CacheDataset(
            data=self.d_train,
            transform=train_transforms,
            cache_num=24,
            cache_rate=1.0,
            num_workers=8,
        )
        self.val_ds = CacheDataset(
            data=self.d_val,
            transform=val_transforms,
            cache_num=6,
            cache_rate=1.0,
            num_workers=4,
        )

    def setup_predict(self):
        predict_transforms = Compose(
            [
                LoadImaged(keys=["CT", "PT"]),
                AddChanneld(keys=["CT", "PT"]),
                # Orientationd(
                #     keys=["CT", "PT"],
                #     axcodes="LPS",
                # ),
                Spacingd(
                    keys=["CT", "PT"],
                    pixdim=self.spacing_size,
                    mode=("bilinear", "bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["CT"],
                    a_min=self.ct_bounds[0],
                    a_max=self.ct_bounds[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                NormalizeIntensityd(keys=["CT", "PT"]),
                ConcatItemsd(keys=["CT", "PT"], name="image"),
                # SelectItemsd(keys=["image"]),
                ToTensord(keys=["image"]),
            ]
        )
        self.pred_ds = Dataset(
            data=self.d_test,
            transform=predict_transforms,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "predict":
            self.setup_predict()
        else:
            self.setup_train_val()

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=1, num_workers=self.num_workers, pin_memory=True
        )
        return val_loader

    def predict_dataloader(self):
        infer_loader = torch.utils.data.DataLoader(
            self.pred_ds, batch_size=1, num_workers=self.num_workers
        )
        return infer_loader
