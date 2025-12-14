# Lesion-Net: Small Lesion Segmentation in Acute Ischemic Stroke
Code for "Lesion-Net: Small Lesion Segmentation in Acute Ischemic Stroke"

![Framework](figures/framework.png)

> This is an official implementation of [Lesion-Net: Small Lesion Segmentation in Acute Ischemic Stroke]() <br>

**Release date:** 16/Dec/2025

## Abstract
This repository implements Lesion-Net, an attention-based segmentation model designed to improve small lesion detection in acute ischemic stroke MRI. The network emphasizes high-resolution features through reduced down-sampling, deeper early encoder stages, and uniform channel widths, enabling better preservation of fine lesion details. Lesion-Net achieves state-of-the-art performance on the ISLES 2022 and JHUS datasets and includes ablation studies demonstrating the impact of each architectural component.

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
ISLES-2022 dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/orvile/isles-2022-brain-stoke-dataset), 

After placing the downloaded 3D volumes in `data/DATASET/3d_data/`, generate 2D slices and patient-wise train/val/test splits using:
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

%## Citation
%If you find this work useful, please consider citing:

%## Acknowledgements
%Parts of this implementation build upon components from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). We thank the authors for making their work publicly available.

