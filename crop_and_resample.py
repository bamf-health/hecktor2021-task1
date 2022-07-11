import argparse
from pathlib import Path

import SimpleITK as sitk
from tqdm.auto import tqdm

import hecktor2021
from hecktor2021.postprocessing.resample_for_submission import resample_to_original


def resample_and_corp(scan, out_dir, bounding_boxes_file):
    img = sitk.ReadImage(str(scan))
    pid = scan.name[:7]
    if "gtvt" in scan.name:
        interp = "nearest"
    else:
        interp = "linear"
    img = resample_to_original(
        img, pid, bounding_boxes_file, force_resolution=[1.0, 1.0, 1.0], interp=interp
    )
    sitk.WriteImage(img, str(out_dir / scan.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hecktor_data_dir",
        type=Path,
        help="path to directory with original hecktor *_ct.nii.g, *_pt.nii.gz, and *_gtvt.nii.gz files",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="path to directory where post processed sitk images are saved with original file names",
    )
    args = parser.parse_args()

    if args.hecktor_data_dir == args.out_dir:
        raise ValueError("hecktor_data_dir and out_dir must not be the same")

    bounding_boxes_file = (
        Path(hecktor2021.__file__).parent / "data" / "bounding_boxes.csv"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scans = list(args.hecktor_data_dir.glob("*.nii.gz"))
    for scan in tqdm(scans):
        resample_and_corp(scan, args.out_dir, bounding_boxes_file)
