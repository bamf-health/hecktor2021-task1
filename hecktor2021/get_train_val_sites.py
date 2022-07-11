from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import KFold


def get_train_val(
    data_path: Path,
    val_site: str,
    heldout_site: str,
    *,
    KFold_number: int = 3,
    fold: int = 0,
    test_path: Path = None,
    psuedo_pred_path: Path = None,
    orig_paths: List[Path] = None,
):
    data_paths = [data_path]
    if test_path:
        data_paths.append(test_path)
    all_subjects = find_subject_paths(data_paths, psuedo_pred_path, orig_paths)

    # only subjects with ground truth data
    subject = all_subjects.dropna(subset=["label"])

    # seperate into train/val/test
    if val_site == "random":
        train = subject[~subject["Subject_ID"].str.contains(heldout_site)]
        test = subject[subject["Subject_ID"].str.contains(heldout_site)]
        kf = KFold(n_splits=KFold_number, shuffle=True, random_state=2)
        train_base = []
        val_base = []
        for train_index, test_index in kf.split(train):
            train_, val_ = subject.iloc[train_index], subject.iloc[test_index]
            train_base.append(train_)
            val_base.append(val_)
        # train, val = train_test_split(subjects_list, test_size=0.25, random_state=1)

        train = train_base[fold]
        val = val_base[fold]
    else:
        train = subject[~subject["Subject_ID"].str.contains(val_site)]
        train = train[~train["Subject_ID"].str.contains(heldout_site)]
        val = subject[subject["Subject_ID"].str.contains(val_site)]
        test = subject[subject["Subject_ID"].str.contains(heldout_site)]

    # add psuedo labeled data to training set
    if psuedo_pred_path:
        # only subjects with pseudo labels
        pl_subjects = all_subjects.dropna(subset=["psuedo_seg"])
        # drop subjects in train or val set, then only options are heldout set or data that doesn't have any ground truth
        pl_subjects = pl_subjects[~pl_subjects["Subject_ID"].isin(train["Subject_ID"])]
        pl_subjects = pl_subjects[~pl_subjects["Subject_ID"].isin(val["Subject_ID"])]
        # drop ground truth paths
        pl_subjects = pl_subjects.drop(columns="label")
        pl_subjects = pl_subjects.rename(columns={"psuedo_seg": "label"})
        # add psuedo label cases to training
        train = train.append(pl_subjects, ignore_index=True)

    return train, val, test


def get_test(
    data_path: Path,
    orig_paths: List[Path] = None,
):
    return find_subject_paths([data_path], orig_paths=orig_paths)


def find_subject_paths(
    data_paths: List[Path], psuedo_pred_path: Path = None, orig_paths: List[Path] = None
):
    # search_string = "_resampled"
    search_string = ""
    subjects: Dict[str, Dict[str, str]] = {}
    for data_path in data_paths:
        add_files_from_dir(
            subjects, data_path, "*_ct" + search_string + ".nii.gz", "CT"
        )
        add_files_from_dir(
            subjects, data_path, "*_pt" + search_string + ".nii.gz", "PT"
        )
        add_files_from_dir(
            subjects, data_path, "*_gtvt" + search_string + ".nii.gz", "label"
        )

    if orig_paths:
        for orig_path in orig_paths:
            add_files_from_dir(
                subjects, orig_path, "*_ct" + search_string + ".nii.gz", "orig_CT"
            )
            add_files_from_dir(
                subjects, orig_path, "*_pt" + search_string + ".nii.gz", "orig_PET"
            )
            add_files_from_dir(
                subjects, orig_path, "*_gtvt" + search_string + ".nii.gz", "orig_seg"
            )

    if psuedo_pred_path:
        add_files_from_dir(
            subjects,
            psuedo_pred_path,
            "*_gtvt" + search_string + ".nii.gz",
            "psuedo_seg",
        )
    df = pd.DataFrame.from_records(list(subjects.values()))
    # drop subjects without input data, could have been added if orig_paths is a superset of data_paths contents
    df.dropna(subset=["CT", "PT"], inplace=True)
    return df


def add_files_from_dir(
    subjects: Dict[str, Dict[str, str]], data_dir: Path, filt: str, key: str
):
    for p in data_dir.rglob(filt):
        subject_ID = p.name.split("_")[0]
        if subject_ID not in subjects:
            subjects[subject_ID] = {"Subject_ID": subject_ID}
        subjects[subject_ID][key] = str(p)
