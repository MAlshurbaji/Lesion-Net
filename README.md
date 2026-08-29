# Lesion-Net: A Lesion-Oriented Hierarchical Transformer for Ischemic Stroke Segmentation in DWI

![Framework](figures/framework.png)

> This is an official implementation of [Lesion-Net: A Lesion-Oriented Hierarchical Transformer for Ischemic Stroke Segmentation in DWI]() <br>

**Release date:** 15/Dec/2025

## Abstract
Accurate segmentation of acute ischemic stroke remains challenging, particularly when lesion burden is low and abnormalities occupy only a small image region. Lesion-Net is a 2D hierarchical transformer framework that adapts spatial resolution, encoder depth, and channel capacity to this setting. It reduces early down-sampling, allocates greater depth to high-resolution stages, and maintains uniform channel widths, together with a lightweight multiscale decoder and feature-fusion head. Lesion-Net achieves DSC scores of 79.82% on ISLES 2022 and 77.14% on JHUS. Controlled ablations support the proposed stage allocation, while burden-stratified evaluation shows its strongest relative advantage in low-burden cases.

## Usage

### Installation
The framework was tested using Python 3.10, PyTorch 2.6, and CUDA 12.4. Ensure that you install all the dependencies listed in `requirements.txt`.

```
conda create -n lesion_net python=3.10
conda activate lesion_net
cd Lesion-Net
pip install -r requirements.txt
```

### Datasets
The ISLES 2022 dataset is publicly available and can be downloaded from [Kaggle](https://www.kaggle.com/datasets/orvile/isles-2022-brain-stoke-dataset). In contrast, the JHUS dataset is a restricted resource and can only be accessed through a formal data request submitted to [ICPSR](https://www.icpsr.umich.edu/web/ICPSR/studies/38464).

After placing the downloaded 3D volumes in `data/isles22/3d_data/`, generate 2D slices and patient-wise train/val/test splits using:
```bash
python make_dataset.py
```

```
data/
└─ isles22/
   ├─ 3d_data/ISLES-2022/
   │  └─ ...
   └─ 2d_data/
      ├─ images/
      │  ├─ train/
      │  ├─ val/
      │  └─ test/
      └─ labels/
         ├─ train/
         ├─ val/
         └─ test/
```

### Training & Evaluation
After setting the parameters in `config/config_train.yaml`, run the following command to train and evaluate the model:
```
python train.py
python evaluate.py
```

<!--
## Citation
If you find this work useful, please consider citing:

```bibtex

```
-->
