import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class BrainSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img = cv2.imread(
            os.path.join(self.image_dir, self.image_filenames[idx]),
            cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.imread(
            os.path.join(self.mask_dir, self.mask_filenames[idx]),
            cv2.IMREAD_GRAYSCALE
        )
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        return torch.tensor(img).unsqueeze(0), torch.tensor(mask).unsqueeze(0)
