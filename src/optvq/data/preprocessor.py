# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from typing import Optional
import numpy as np

from torchvision import transforms
import albumentations as A

BICUBIC = transforms.InterpolationMode.BICUBIC

normalize_params = {
    "plain": {"mean": (0.5,), "std": (0.5,)},
    "cnn": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "clip": {"mean": (0.48145466, 0.4578275, 0.40821073), "std": (0.26862954, 0.26130258, 0.27577711)}
}

recover_map_dict = {
    "plain": transforms.Normalize(
        mean=(-1,), std=(2,)
    ),
    "cnn": transforms.Normalize(
        mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
        std=(1/0.229, 1/0.224, 1/0.225)
    ),
    "clip": transforms.Normalize(
        mean=(-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711),
        std=(1/0.26862954, 1/0.26130258, 1/0.27577711)
    )
}

def get_recover_map(name: str):
    return recover_map_dict[name]

###########################################
# Preprocessor
###########################################

def plain_preprocessor(resize: Optional[int] = 32):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize(resize),
    ])

def imagenet_preprocessor(resize: Optional[int] = 256, is_train: bool = True):
    if is_train:
        # augmentation v1
        # transform = A.Compose([
        #     A.SmallestMaxSize(max_size=resize),
        #     A.RandomCrop(height=resize, width=resize),
        #     A.HorizontalFlip(p=0.5),
        # ])

        # augmentation v2
        transform = A.Compose([
            A.SmallestMaxSize(max_size=resize),
            A.RandomResizedCrop(width=resize, height=resize, scale=(0.2, 1.0)),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=7, p=0.5),
            A.HorizontalFlip(p=0.5),
        ])
    else:
        transform = A.Compose([
            A.SmallestMaxSize(max_size=resize),
            A.CenterCrop(height=resize, width=resize),
        ])
    return transform