# BAMF Hecktor 2021

BAMF Health's entry to the [Hecktor 2021 competition](https://www.aicrowd.com/challenges/miccai-2021-hecktor).

## Setup

1. Clone repo
2. install package

```sh
poetry install
pre-commit install
poetry shell
# install a different version of torch if you need cuda 11+ support
pip3 install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Get HECKTOR 2021 Data

Downlad the HECKTOR 2021 dataset and extract it in a directory with the the following structure:
Set HECKTOR_BASE_DIR as an environmental variable for easier use with subsequent example code

```
HECKTOR_BASE_DIR/
    hecktor2021_bbox_testing.csv
    hecktor2021_bbox_training.csv
    hecktor2021_patient_endpoint_training.csv
    hecktor2021_patient_info_testing.csv
    hecktor2021_patient_info_training.csv
    hecktor_nii/
        CHGJ007_ct.nii.gz
        CHGJ007_gtvt.nii.gz
        CHGJ007_pt.nii.gz
        ...
    hecktor_nii_test/
        CHUP025_ct.nii.gz
        CHUP025_pt.nii.gz
        ...
```

### Resample HECKTOR data

Use the `crop_and_resample.py` script to crop and resample the HECKTOR data. Save to `hecktor_nii_resample_cropped` directory next to `hecktor_nii` directory for ease of use. Also resample to test data in `hecktor_nii_test` and save to `hecktor_nii_test_resample_cropped` directory.

```sh
python crop_and_resample.py $HECKTOR_BASE_DIR/hecktor_nii $HECKTOR_BASE_DIR/hecktor_nii_resample_cropped $HECKTOR_BASE_DIR/hecktor2021_bbox_training.csv
python crop_and_resample.py $HECKTOR_BASE_DIR/hecktor_nii_test $HECKTOR_BASE_DIR/hecktor_nii_test_resample_cropped $HECKTOR_BASE_DIR/hecktor2021_bbox_testing.csv
```

## Run

### 5 fold splits

Five fold splits have been created and stored in medical decathelon style json files under the `splits` directory.

### Train

The code base allows for many settings and models, but train all folds of the DynUnet and ResDynUnet models to match submitted 2021 results.
You can use `python train.py --help` for a complete list of options.

```sh
python train.py --datasplit_json splits/dataset_0.json --datasplit_base_dir $HECKTOR_BASE_DIR --model_type DynUnet --max_epochs 1000 --experiment_suffix fold0
python train.py --datasplit_json splits/dataset_1.json --datasplit_base_dir $HECKTOR_BASE_DIR --model_type DynUnet --max_epochs 1000 --experiment_suffix fold1
python train.py --datasplit_json splits/dataset_2.json --datasplit_base_dir $HECKTOR_BASE_DIR --model_type DynUnet --max_epochs 1000 --experiment_suffix fold2
python train.py --datasplit_json splits/dataset_3.json --datasplit_base_dir $HECKTOR_BASE_DIR --model_type DynUnet --max_epochs 1000 --experiment_suffix fold3
python train.py --datasplit_json splits/dataset_4.json --datasplit_base_dir $HECKTOR_BASE_DIR --model_type DynUnet --max_epochs 1000 --experiment_suffix fold4
python train.py --datasplit_json splits/dataset_0.json --datasplit_base_dir $HECKTOR_BASE_DIR --model_type ResDynUnet --max_epochs 1000 --experiment_suffix fold0
python train.py --datasplit_json splits/dataset_1.json --datasplit_base_dir $HECKTOR_BASE_DIR --model_type ResDynUnet --max_epochs 1000 --experiment_suffix fold1
python train.py --datasplit_json splits/dataset_2.json --datasplit_base_dir $HECKTOR_BASE_DIR --model_type ResDynUnet --max_epochs 1000 --experiment_suffix fold2
python train.py --datasplit_json splits/dataset_3.json --datasplit_base_dir $HECKTOR_BASE_DIR --model_type ResDynUnet --max_epochs 1000 --experiment_suffix fold3
python train.py --datasplit_json splits/dataset_4.json --datasplit_base_dir $HECKTOR_BASE_DIR --model_type ResDynUnet --max_epochs 1000 --experiment_suffix fold4
```

### Inference

Run inference on test data for all models and folds

```sh
python infer.py experiments/DynUnet_fold0/models/DynUnet_fold0-v1.ckpt $HECKTOR_BASE_DIR/pred/DynUnet_fold0 --softmax --predict_aug
python infer.py experiments/DynUnet_fold0/models/DynUnet_fold1-v1.ckpt $HECKTOR_BASE_DIR/pred/DynUnet_fold1 --softmax --predict_aug
python infer.py experiments/DynUnet_fold0/models/DynUnet_fold2-v1.ckpt $HECKTOR_BASE_DIR/pred/DynUnet_fold2 --softmax --predict_aug
python infer.py experiments/DynUnet_fold0/models/DynUnet_fold3-v1.ckpt $HECKTOR_BASE_DIR/pred/DynUnet_fold3 --softmax --predict_aug
python infer.py experiments/DynUnet_fold0/models/DynUnet_fold4-v1.ckpt $HECKTOR_BASE_DIR/pred/DynUnet_fold4 --softmax --predict_aug
python infer.py experiments/ResDynUnet_fold0/models/ResDynUnet_fold0-v1.ckpt $HECKTOR_BASE_DIR/pred/ResDynUnet_fold0 --softmax --predict_aug
python infer.py experiments/ResDynUnet_fold0/models/ResDynUnet_fold1-v1.ckpt $HECKTOR_BASE_DIR/pred/ResDynUnet_fold1 --softmax --predict_aug
python infer.py experiments/ResDynUnet_fold0/models/ResDynUnet_fold2-v1.ckpt $HECKTOR_BASE_DIR/pred/ResDynUnet_fold2 --softmax --predict_aug
python infer.py experiments/ResDynUnet_fold0/models/ResDynUnet_fold3-v1.ckpt $HECKTOR_BASE_DIR/pred/ResDynUnet_fold3 --softmax --predict_aug
python infer.py experiments/ResDynUnet_fold0/models/ResDynUnet_fold4-v1.ckpt $HECKTOR_BASE_DIR/pred/ResDynUnet_fold4 --softmax --predict_aug
```

### Ensemble results

```sh
python ensemble.py  $HECKTOR_BASE_DIR/ensemble  $HECKTOR_BASE_DIR/pred/*
```
