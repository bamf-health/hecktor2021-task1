from monai.networks.blocks.squeeze_and_excitation import SEBottleneck
from monai.networks.nets import UNETR as monai_UNETR
from monai.networks.nets import AHNet, SegResNet, SegResNetVAE, SENet

from hecktor2021.models.dyn_unet import create_DynUNet

VALID_MODELS = [
    "DynUnet",
    "ResDynUnet",
]


class HecktorBackbone:
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Backbone model")
        parser.add_argument(
            "--model_type",
            type=str,
            default="DynUnet",
            choices=VALID_MODELS,
            help="Choose a model",
        )
        parser.add_argument(
            "--dimensions",
            type=int,
            choices=[2, 3],
            default=3,
            help="model input spacial dimensions",
        )
        parser.add_argument(
            "--in_channels", type=int, default=2, help="model input channels"
        )
        parser.add_argument(
            "--output_ch",
            type=int,
            default=2,
            help="model output channels, including the background as a channel",
        )
        return parent_parser

    @staticmethod
    def build_model(
        model_type,
        img_ch=None,
        dim=None,
        output_ch=None,
        t=None,
        patch_size=(128, 128, 128),
        spacing=None,
        **kwargs,
    ):
        """Build generator and discriminator."""
        if model_type == "AHNet":
            if dim == 2:
                upsampling_type = "bilinear"
            else:
                upsampling_type = "trilinear"
            unet = AHNet(
                layers=(2, 3, 2, 3),
                spatial_dims=dim,
                in_channels=img_ch,
                out_channels=1,
                psp_block_num=4,
                upsample_mode=upsampling_type,
                pretrained=False,
                progress=False,
            )
        elif model_type == "SegResNet":
            unet = SegResNet(
                spatial_dims=dim,
                init_filters=16,
                in_channels=img_ch,
                out_channels=output_ch,
                dropout_prob=0.1,
                norm_name="group",
                num_groups=8,
                use_conv_final=True,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                upsample_mode="deconv",
            )
        elif model_type == "SegResNetVAE":
            unet = SegResNetVAE(
                patch_size,
                vae_estimate_std=False,
                vae_default_std=0.3,
                vae_nz=256,
                spatial_dims=dim,
                init_filters=8,
                in_channels=img_ch,
                out_channels=output_ch,
                dropout_prob=None,
                norm_name="group",
                num_groups=8,
                use_conv_final=True,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                upsample_mode="nontrainable",
            )
        elif model_type == "SENet":
            unet = SENet(
                spatial_dims=dim,
                in_channels=img_ch,
                block=SEBottleneck,
                layers=(3, 8, 36, 3),
                groups=64,
                reduction=16,
                dropout_prob=0.2,
                dropout_dim=1,
                inplanes=128,
                downsample_kernel_size=3,
                input_3x3=True,
                num_classes=1,
            )  # https://docs.monai.io/en/latest/_modules/monai/networks/nets/senet.html
        elif model_type == "DynUnet":
            if spacing is None:
                raise ValueError("spacing must be provided for DynUnet")
            unet = create_DynUNet(dim, img_ch, output_ch, patch_size, spacing)
        elif model_type == "ResDynUnet":
            if spacing is None:
                raise ValueError("spacing must be provided for DynUnet")
            unet = create_DynUNet(dim, img_ch, output_ch, patch_size, spacing, res=True)
        elif model_type == "monai_UNETR":
            assert dim == 3
            unet = monai_UNETR(
                in_channels=img_ch,
                out_channels=output_ch,
                img_size=patch_size,
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.0,
            )
        else:
            raise RuntimeError("Unknown model ", model_type)

        return unet
