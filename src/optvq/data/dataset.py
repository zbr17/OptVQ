# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .preprocessor import normalize_params

class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None, convert_to_numpy: bool = True, post_normalize: str = "plain"):
        self.root = root
        self.transform = transform
        self.convert_to_numpy = convert_to_numpy
        self.post_normalize = transforms.Normalize(
            **normalize_params[post_normalize]
        )

        # find classes
        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # make dataset
        self.samples = []
        self.extensions = []
        for target_class in sorted(class_to_idx.keys()):
            class_idx = class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for fname in sorted(os.listdir(target_dir)):
                path = os.path.join(target_dir, fname)
                item = (path, class_idx)
                self.samples.append(item)
                ext = path.split(".")[-1]
                if ext not in self.extensions:
                    self.extensions.append(ext)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.convert_to_numpy:
            image = np.array(image).astype("uint8")
        # image augmentation
        image = self.transform(image=image)["image"]
        # to tensor and normalize
        image = (image / 255).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = self.post_normalize(image)
        return image, label
