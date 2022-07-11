# imports
import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage
import skimage.measure

import hecktor2021


# soft max conversion of predictions
def softmax_conversion(scan_file: Path):
    """
    scan_file: file path to file that needs to be converted as a softmax
     * pkl, nii.gz, npz file for each sample
    """
    pred_soft_max = scan_file.parent / (scan_file.stem.split(".")[0] + ".npz")
    metadata_file = scan_file.parent / (scan_file.stem.split(".")[0] + ".pkl")
    with np.load(pred_soft_max) as pred_softmax_fp:
        pred_softmax = pred_softmax_fp["softmax"]
        pred_softmax = pred_softmax[1].astype(
            "float"
        )  # just get predictions for tumor class

    with metadata_file.open("rb") as fp:
        metadata = pickle.load(fp)
    pred_sitk = sitk.ReadImage(str(scan_file))
    pred_arr = sitk.GetArrayFromImage(pred_sitk)
    if pred_arr.shape != pred_softmax.shape:  # make sure shapes match after conversion
        crop_bbox = np.array(metadata["crop_bbox"])
        print("shape mismatch: ", pred_arr.shape, pred_softmax.shape)
        print("inserting softmax into: ", crop_bbox)
        new_img = np.zeros_like(pred_arr)
        new_img[
            crop_bbox[0, 0] : crop_bbox[0, 1],
            crop_bbox[1, 0] : crop_bbox[1, 1],
            crop_bbox[2, 0] : crop_bbox[2, 1],
        ] = pred_softmax
        pred_softmax = new_img

    pred_softmax_sitk = sitk.GetImageFromArray(pred_softmax)
    pred_softmax_sitk.CopyInformation(pred_sitk)

    return pred_softmax_sitk


# get brain mask
def get_brain_mask(pt_scan: Path):
    """
    pt_scan: file path to pt scan to obtain the brain mask from
        uses hecktor_nii_resampled_full to obtain full mask of the brain
    """
    # load pt scan
    pt_sitk = sitk.ReadImage(str(pt_scan))
    pt = sitk.GetArrayFromImage(pt_sitk)
    threshold = np.mean(pt) + 3 * np.std(pt)  # set threshold to be 3x std dev.
    pt_thres = pt >= threshold  # convert pt to binary based on threshold
    labels = skimage.measure.label(
        pt_thres, background=0, connectivity=None
    )  # get labels
    props = skimage.measure.regionprops(labels)

    # get largest label = brain
    sorted_labels = sorted(props, key=lambda x: x.area)
    largest_label = sorted_labels[-1].label
    brain = labels == largest_label
    brain = brain * 1

    brain_mask = sitk.GetImageFromArray(brain)
    brain_mask.CopyInformation(pt_sitk)

    return brain_mask


# resample to original resolution and bbox
def resample_to_original(
    img,
    pid: str,
    bounding_boxes_file: Path,
    orig_resolution_file: Path = None,
    force_resolution=None,
    interp="linear",
):
    """
    img: sitk image that needs to be resampled to the bbox and orginal resolution
    pid: patient id
    bounding_boxes_file: defines the coordinates of the boundary box
    orig_resolution_file: original CT resolution
    """
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index("PatientID")
    if orig_resolution_file:
        res_df = pd.read_csv(orig_resolution_file)
        res_df = res_df.set_index("PatientID")
        resampling = [
            res_df.loc[pid, "Resolution_x"],
            res_df.loc[pid, "Resolution_y"],
            res_df.loc[pid, "Resolution_z"],
        ]
    elif force_resolution:
        resampling = force_resolution
    else:
        raise RuntimeError(
            "Either orig_resolution_file or force_resolution must be set"
        )

    # resample one patient at a time
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)
    bb = np.array(
        [
            bb_df.loc[pid, "x1"],
            bb_df.loc[pid, "y1"],
            bb_df.loc[pid, "z1"],
            bb_df.loc[pid, "x2"],
            bb_df.loc[pid, "y2"],
            bb_df.loc[pid, "z2"],
        ]
    )
    size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])  # sitk is so stupid
    if interp == "nearest":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interp == "linear":
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        raise RuntimeError(f"unknown interpretor {interp}")
    img = resampler.Execute(img)

    return img


# apply brain mask
def remove_brain(pred, brain):
    """
    pred: resampled prediction as an sitk image
    brain: resampled brain mask as an sitk image

    """
    # convert incoming images back to numpy array
    brain_mask = sitk.GetArrayFromImage(brain)
    pred_np = sitk.GetArrayFromImage(pred)

    # remove brain from AI prediction by appling brain label
    brainless_pred = np.copy(pred_np)
    brainless_pred[brain_mask] = 0

    # convert back to sitk image
    brainless_pred_sitk = sitk.GetImageFromArray(brainless_pred)
    brainless_pred_sitk.CopyInformation(pred)

    return brainless_pred_sitk


def convert_dtype(img: sitk.Image, dtype="uint8"):
    u32_img = sitk.GetImageFromArray(sitk.GetArrayFromImage(img).astype(dtype))
    u32_img.CopyInformation(img)
    return u32_img


# single post processing
def post_processing_pred_single(
    pt_file: Path,
    pred_file: Path,
    bounding_boxes_file: Path,
    orig_resolution_file: Path,
    out_dir: Path,
    threshold: float = None,
):
    """
    pt_file: path to single pt file
    pred_file: path to single prediction file
    bounding_boxes_file: defines the coordinates of the boundary box
    orig_resolution_file: original CT resolution
    out_dir: path to directory where post processed sitk images are saved as .nii.gz files
    """
    assert pt_file.name[:7] == pred_file.name[:7]
    pid = pred_file.name[:7]  # get patient id
    pred_softmax = softmax_conversion(pred_file)
    # pred_softmax = sitk.ReadImage(str(pred_file))
    brain_mask = get_brain_mask(pt_file)

    # resample both brain and prediction to match bbox and resolution
    brain_resample = resample_to_original(
        brain_mask, pid, bounding_boxes_file, orig_resolution_file
    )
    pred_resample = resample_to_original(
        pred_softmax, pid, bounding_boxes_file, orig_resolution_file
    )

    # apply brain mask to the predictions
    brainless_pred = remove_brain(pred_resample, brain_resample)
    # brainless_pred = pred_resample

    # threshold
    if threshold:
        brainless_pred = brainless_pred > threshold
        brainless_pred = convert_dtype(brainless_pred)

    # write to file in format pid.nii.gz
    out_file = out_dir / (pid + ".nii.gz")
    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(brainless_pred, str(out_file))


# multi postprocessing
def post_processing_pred_multiple(
    pt_files_dir: Path,
    pred_files_dir: Path,
    bounding_boxes_file: Path,
    orig_resolution_file: Path,
    out_dir: Path,
    threshold: float = None,
):
    """
    pt_files_dir: path to directory with pt files
    pred_files_dir: path to directory with prediction files
    bounding_boxes_file: defines the coordinates of the boundary box
    orig_resolution_file: original CT resolution
    out_dir: path to directory where post processed sitk images are saved as .nii.gz files
    threashold: Threshold softmax predictions at this value
    """
    pt_scans = sorted(list(pt_files_dir.glob("*.nii.gz")))
    pred_scans = sorted(list(pred_files_dir.glob("*.nii.gz")))

    # filter pt scans that do not have corresponding prediction
    pt_scan_dic = {k.name[:7]: k for k in pt_scans}
    pt_scans = [pt_scan_dic[k.name[:7]] for k in pred_scans]
    assert len(pt_scans) == len(pred_scans)

    args = [
        (pt, pred, bounding_boxes_file, orig_resolution_file, out_dir, threshold)
        for pred, pt in zip(pred_scans, pt_scans)
    ]

    with Pool(24) as p:
        p.starmap(post_processing_pred_single, args)


if __name__ == "__main__":
    bounding_boxes_file = (
        Path(hecktor2021.__file__).parent / "data" / "bounding_boxes.csv"
    )
    orig_resolution_file = (
        Path(hecktor2021.__file__).parent / "data" / "original_resolution_ct.csv"
    )
    parser = argparse.ArgumentParser(description="Post processing of AI predictions")
    parser.add_argument(
        "pt_files_dir",
        type=Path,
        help="path to directory with pt files",
    )
    parser.add_argument(
        "pred_files_dir",
        type=Path,
        help="path to directory with prediction files",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="path to directory where post processed sitk images are saved as .nii.gz files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold softmax predictions at this value",
    )
    parser.add_argument(
        "--bounding_boxes",
        type=Path,
        default=bounding_boxes_file,
        help="defines the coordinates of the boundary box for samples",
    )
    parser.add_argument(
        "--orig_resolution",
        type=Path,
        default=orig_resolution_file,
        help="original CT resolution for samples",
    )
    args = parser.parse_args()

    post_processing_pred_multiple(
        args.pt_files_dir,
        args.pred_files_dir,
        args.bounding_boxes,
        args.orig_resolution,
        args.out_dir,
        args.threshold,
    )
