import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm

import hecktor2021
from hecktor2021.postprocessing.resample_for_submission import resample_to_original


def get_fold_predictions(prediciton_dirs: List[Path]):
    all_folds: List[List[Path]] = []
    for prediciton_dir in prediciton_dirs:
        all_folds.append(sorted(list(prediciton_dir.glob("*.nii.gz"))))

    # sanity checks
    for fold_predictions in all_folds[1:]:
        if len(fold_predictions) != len(all_folds[0]):
            raise RuntimeError("All predictions must have the same number of files")

        for f1, f2 in zip(all_folds[0], fold_predictions):
            if f1.name != f2.name:
                raise RuntimeError("All predictions must have the same names")

    return all_folds


def ensemble_predictions(all_folds: List[List[Path]], output_dir: Path):
    threshold = 0.2
    for samples in tqdm(zip(*all_folds), total=len(all_folds[0])):
        x: Optional[np.ndarray] = None
        ref_img = None
        pid = samples[0].name[:7]
        for fold in samples:
            s_img = sitk.ReadImage(str(fold))
            s_img = resample_to_original(
                s_img, pid, bounding_boxes_file, orig_resolution_file, interp="linear"
            )
            if x is None:
                x = sitk.GetArrayFromImage(s_img)
                ref_img = s_img
            else:
                x = x + sitk.GetArrayFromImage(s_img)
        assert x is not None
        x = x / len(samples)
        x = x > threshold
        x = x.astype(np.uint8)
        seg_img = sitk.GetImageFromArray(x)
        seg_img.CopyInformation(ref_img)

        #     labels_count = skilabel(x,return_num=True)[1]
        #     shape_stats = sitk.LabelShapeStatisticsImageFilter()
        #     shape_stats.Execute(sitk.ConnectedComponent(seg_img))
        #     print(f"{pid} sitk labels {shape_stats.GetLabels()} vs {labels_count}")
        labeled_segmentation = sitk.ConnectedComponent(seg_img)
        labeled_segmentation = sitk.RelabelComponent(
            labeled_segmentation, minimumObjectSize=1, sortByObjectSize=True
        )
        binary_segmentation = labeled_segmentation == 1

        out_file = output_dir / (pid + ".nii.gz")
        sitk.WriteImage(binary_segmentation, str(out_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", type=Path, help="Path to output directory for ensembled results"
    )
    parser.add_argument(
        "fold_prediction_dirs",
        type=Path,
        nargs="+",
        help="Path to prediction directories",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bounding_boxes_file = (
        Path(hecktor2021.__file__).parent / "data" / "bounding_boxes.csv"
    )
    orig_resolution_file = (
        Path(hecktor2021.__file__).parent / "data" / "original_resolution_ct.csv"
    )

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(args.output_dir.glob("*.nii.gz")) > 0 and not args.overwrite:
        raise RuntimeError(
            "Output directory must be empty, unless 'overwrite' flag is set"
        )

    all_folds = get_fold_predictions(args.fold_prediction_dirs)
    ensemble_predictions(all_folds, args.output_dir)
