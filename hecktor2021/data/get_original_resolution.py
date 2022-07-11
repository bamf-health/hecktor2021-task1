import glob
import logging
from pathlib import Path

import click
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


@click.command()
@click.argument("input_folder", type=click.Path(exists=True), default="data/processed")
@click.argument(
    "output_file", type=click.Path(), default="data/original_resolution_ct_new.csv"
)
@click.option("--extension", type=click.STRING, default=".nii.gz")
def main(input_folder, output_file, extension):
    """Command Line Inteface used to generate a csv file containing the
       original voxel spacing.
    Args:
        input_folder (str): Path containing the NIFTI images in the original
                            resolution.
        output_file (str): Path where to store the csv.
        extension (str): String containing the extension of the NIFTI files ('.nii' or '.nii.gz')
    """

    resolution_dict = pd.DataFrame()
    for f in tqdm(list(Path(input_folder).rglob("*_ct" + extension))):
        patient_name = f.name.split("_")[0]
        sitk_image = sitk.ReadImage(str(f))
        px_spacing = sitk_image.GetSpacing()
        resolution_dict = resolution_dict.append(
            {
                "PatientID": patient_name,
                "Resolution_x": px_spacing[0],
                "Resolution_y": px_spacing[1],
                "Resolution_z": px_spacing[2],
            },
            ignore_index=True,
        )

    resolution_dict.to_csv(output_file)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)
    main()
