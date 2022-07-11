"""Modified from https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_pipeline/create_network.py"""
import torch
from monai.networks.nets import DynUNet

__all__ = ["create_DynUNet"]


def _get_kernels_strides(patch_size, spacing):
    sizes, spacings = patch_size, spacing
    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def create_DynUNet(dim, in_channels, n_class, patch_size, spacing, res=False):
    kernels, strides = _get_kernels_strides(patch_size, spacing)
    deep_supr_num = len(strides) - 2

    net = DynUNet(
        spatial_dims=dim,
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=deep_supr_num,
        res_block=res,
    )

    return net


def compute_pred_loss(criterion, preds, targets):
    """Compute the loss across deep supervision heads"""
    if len(preds.size()) == len(targets.size()):
        # no deepsupervision mode
        loss = criterion(preds, targets)
    else:
        # deep supervision mode, need to unbind feature maps first.
        # deeper feature map losses are weighted less
        preds_list = torch.unbind(preds, dim=1)
        loss = sum(0.5 ** i * criterion(p, targets) for i, p in enumerate(preds_list))
    return loss


def remove_deep_supervision(preds, targets):
    """Compute the loss across deep supervision heads"""
    if len(preds.size()) == len(targets.size()):
        # no deepsupervision mode
        return preds
    else:
        # deep supervision mode
        return preds[:, 0, ...]
