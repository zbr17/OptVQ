# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from typing import Union

import torch
from torch.utils.data import Subset

def maybe_get_subset(dataset, subset_size: Union[int, float] = None, num_data_repeat: int = None):
    """
    num_data_repeat is aimed at avoiding the consuming loading time for the small dataset
    """
    if subset_size is None:
        return dataset
    else:
        if subset_size < 1.0:
            subset_size = len(dataset) * subset_size
        subset_size = int(subset_size)
        selected_indices = torch.randperm(len(dataset))[:subset_size]
        if num_data_repeat is not None:
            selected_indices = selected_indices.repeat(num_data_repeat)
        return Subset(dataset, selected_indices)

class LoaderWrapper:
    """
    write a dataloader class, given total steps, recursively loading data
    """
    def __init__(self, loader, total_iterations: int = None):
        self.loader = loader
        self.total_iterations = total_iterations if total_iterations is not None else len(loader)
    
    def __iter__(self):
        self.generator = iter(self.loader)
        self.counter = 0
        return self
    
    def __next__(self):
        if self.counter >= self.total_iterations:
            self.counter = 0
            raise StopIteration
        else:
            self.counter += 1
            try:
                return next(self.generator)
            except StopIteration:
                self.generator = iter(self.loader)
                return next(self.generator)