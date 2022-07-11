import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from typing import Sequence

import pytorch_lightning
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

from hecktor2021.backbone import HecktorBackbone
from hecktor2021.models.dyn_unet import compute_pred_loss, remove_deep_supervision


class HecktorTask(pytorch_lightning.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HecktorTask")
        parser.add_argument(
            "--loss",
            default="nnunet",
            choices=["nnunet", "adamw"],
            help="Loss function to use",
        )
        parser.add_argument(
            "--amsgrad",
            action="store_true",
            help="if using adamw loss, set this to use amsgrad version",
        )
        parser.add_argument("--lr", type=float, default=0.002)
        parser.add_argument(
            "--beta1", type=float, default=0.9, help="momentum1 in AdamW"
        )
        parser.add_argument(
            "--beta2", type=float, default=0.999, help="momentum2 in AdamW"
        )
        parser.add_argument(
            "--predict_aug",
            action="store_true",
            help="Use test time augmentation flips",
        )
        parent_parser = HecktorBackbone.add_model_specific_args(parent_parser)
        return parent_parser

    def __init__(
        self,
        model_type: str,
        *,
        in_channels: int = None,
        output_ch: int = None,
        patch_size: Sequence[int] = (144, 144, 144),
        dimensions: int = None,
        t: int = None,
        loss: str = "nnunet",
        amsgrad: bool = False,
        lr=1e-3,
        beta1: float = 0.5,
        beta2: float = 0.999,
        spacing_size: Sequence[float] = None,
        predict_aug=False,
        **kwargs,
    ):
        assert in_channels
        assert output_ch

        super().__init__()
        self.model_type = model_type
        self.patch_size = patch_size
        self.predict_aug = predict_aug

        # Optimizer Hyper-parameters
        self.loss_name = loss
        self.amsgrad = amsgrad
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self._model = HecktorBackbone.build_model(
            model_type,
            in_channels,
            dimensions,
            output_ch,
            t,
            patch_size=patch_size,
            spacing=spacing_size,
        )
        self.criterion = DiceCELoss(
            include_background=False, to_onehot_y=True, softmax=True, batch=True
        )

        self.post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=output_ch)
        self.post_label = AsDiscrete(to_onehot=True, n_classes=output_ch)
        self.metric_dice = DiceMetric(include_background=False)
        self.metric_haus = HausdorffDistanceMetric(
            include_background=False, percentile=95
        )

        self.best_val_dice = 0
        self.best_val_haus = None
        self.best_val_epoch = 0
        self.save_hyperparameters()

    def forward(self, x):
        if self.model_type == "SegResNetVAE":
            y = self._model(x)[0]
        else:
            y = self._model(x)
        return y

    def configure_optimizers(self):
        # from UNETR
        if self.loss_name == "adamw":
            return torch.optim.AdamW(
                self._model.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                weight_decay=1e-5,
                amsgrad=self.amsgrad,
            )

        # from nnUnet
        elif self.loss_name == "nnunet":
            optimizer = torch.optim.SGD(
                self._model.parameters(),
                lr=0.01,  # self.lr,
                momentum=0.99,
                nesterov=True,
            )

            def poly_decay(epoch):
                return (1 - epoch / self.hparams.max_epochs) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=poly_decay
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            raise RuntimeError("Unknown loss")

    def _compute_metrics(self, outputs, labels, prefix=""):
        outputs = torch.stack([self.post_pred(i) for i in decollate_batch(outputs)])
        labels = torch.stack([self.post_label(i) for i in decollate_batch(labels)])
        dice = self.metric_dice(y_pred=outputs, y=labels)
        dice = self.metric_dice.aggregate()
        self.metric_dice.reset()
        metrics = {prefix + "dice": dice}
        if not self.training:  # hausdorff distance just takes so long
            haus = self.metric_haus(y_pred=outputs, y=labels)
            haus = self.metric_haus.aggregate()
            self.metric_haus.reset()
            metrics[prefix + "haus"] = haus
        return metrics

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        if self.model_type in ("DynUnet", "ResDynUnet"):
            # special case to handle deep supervision
            loss = compute_pred_loss(self.criterion, outputs, labels)
            outputs = remove_deep_supervision(outputs, labels)
        else:
            loss = self.criterion(outputs, labels)

        # TODO: what about using torchmetrics package instead?
        metrics = self._compute_metrics(outputs, labels, prefix="train/")
        metrics["train/loss"] = loss
        self.log_dict(metrics, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, self.patch_size, sw_batch_size, self.forward
        )
        loss = self.criterion(outputs, labels)

        metrics = self._compute_metrics(outputs, labels, prefix="val/")
        metrics["val/loss"] = loss
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, self.patch_size, sw_batch_size, self.forward
        )
        loss = self.criterion(outputs, labels)

        metrics = self._compute_metrics(outputs, labels, prefix="test/")
        metrics["test/loss"] = loss
        self.log_dict(metrics)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        images = batch["image"]
        sw_batch_size = 4
        pred = sliding_window_inference(
            images, self.patch_size, sw_batch_size, self.forward
        )
        pred = torch.softmax(pred, dim=1)
        cnt = 1
        if self.predict_aug:
            for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                flip_inputs = torch.flip(images, dims=dims)
                flip_pred = torch.flip(
                    sliding_window_inference(
                        flip_inputs, self.patch_size, sw_batch_size, self.forward
                    ),
                    dims=dims,
                )
                flip_pred = torch.softmax(flip_pred, dim=1)
                pred += flip_pred
                cnt += 1
        pred = pred / cnt
        return pred.detach().cpu()
        batch["pred"] = pred
        # val_post_transforms = Compose(
        #     [
        #         EnsureTyped(keys="pred"),
        #         Activationsd(keys="pred", sigmoid=True),
        #         AsDiscreted(keys="pred", threshold_values=True),
        #         # KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        #         SaveImaged(keys="pred", meta_keys="CT_meta_dict", output_dir="./runs/"),
        #     ]
        # )
        # val_post_transforms(batch)

        return batch
