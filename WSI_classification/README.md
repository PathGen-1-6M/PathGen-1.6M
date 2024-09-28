
# Dataset Preparation

We provide a part of the extracted features to reimplement our results. 


## Camelyon16 Dataset (20× magnification)

| Model        | Download Link                                   |
| ------------ | ----------------------------------------------- |
| SSL ViT      | [Download](https://pan.quark.cn/s/6ea54bfa0e72) |
| PathGen-CLIP | [Download](https://pan.quark.cn/s/62fe3dc65291) |

## Bracs Dataset

| Model        | Download Link                                   |
| ------------ | ----------------------------------------------- |
| SSL ViT      | [Download](https://pan.quark.cn/s/3c8c1ffce517) |
| PathGen-CLIP | [Download](https://pan.quark.cn/s/62fe3dc65291) |

For your own dataset, you can modify and run [Step1_create_patches_fp.py](Step1_create_patches_fp.py) and [Step2_feature_extract.py](Step2_feature_extract.py). More details about this file can refer [**CLAM**](https://github.com/mahmoodlab/CLAM/).
Note that we recommend extracting features using SSL pretrained method. Our code using the checkpoints provided by [Benchmarking Self-Supervised Learning on Diverse Pathology Datasets](https://openaccess.thecvf.com/content/CVPR2023/html/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.html)

# Training

For the ABMIL (baseline), you can run [Step3_WSI_classification_ACMIL.py](Step3_WSI_classification_ACMIL.py) and set n_token=1 n_masked_patch=0 mask_drop=0

```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification_ACMIL.py --seed 4 --wandb_mode online --arch ga --n_token 1 --n_masked_patch 0 --mask_drop 0 --config config/bracs_natural_supervised_config.yml
```

For our ACMIL, you can run [Step3_WSI_classification_ACMIL.py](Step3_WSI_classification_ACMIL.py) and set n_token=5 n_masked_patch=10 mask_drop=0.6

```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification_ACMIL.py --seed 4 --wandb_mode online --arch ga --n_token 5 --n_masked_patch 10 --mask_drop 0.6 --config config/bracs_natural_supervised_config.yml
```


