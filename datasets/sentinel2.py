import os
import pandas as pd
import numpy as np
import warnings
import random
from glob import glob
import json
from typing import Any, Optional, List

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
# import rasterio
# from rasterio import logging

from pathlib import Path

import skimage.io as io

class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)

class EuroSat(SatelliteDataset):
    # mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
    #         1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
    #         1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    # std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
    #        948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
    #        1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

    mean = [1370.19151926, 1184.3824625 , 1120.77120066]

    std = [633.15169573,  650.2842772 ,  712.12507725]

    mean_b = [1136.25]

    std_b = [963.25]

    def __init__(self, file_path, transform, masked_bands=None, dropped_bands=None):
        """
        Creates dataset for multi-spectral single image classification for EuroSAT.
        :param file_path: path to txt file containing paths to image data for EuroSAT.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(13)
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
        self.img_paths = [row.split()[0] for row in data]
        self.labels = [int(row.split()[1]) for row in data]

        self.transform = transform

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.img_paths)

    def open_image(self, img_path):
        # with rasterio.open('/home/ps/Documents/data/'+img_path) as data:
        #     img = data.read()  # (c, h, w)
        img = io.imread('data/'+ img_path)

        # kid = (img - img.min(axis=(0, 1), keepdims=True))
        # mom = (img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True))
        # img = kid / (mom+1e-10)
        # # return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = self.open_image(img_path)  # (h, w, c)
        if self.masked_bands is not None:
            img[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        img_as_tensor = self.transform(img)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        return img_as_tensor, label