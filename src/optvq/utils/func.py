# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from typing import Tuple, Union, Iterable
from omegaconf import OmegaConf
import os

import torch
import torch.distributed as dist

def dist_all_gather(x):
    tensor_list = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, x)
    x = torch.cat(tensor_list, dim=0)
    return x

def any_2tuple(data: Union[int, Tuple[int]]) -> Tuple[int]:
    if isinstance(data, int):
        return (data, data)
    elif isinstance(data, Iterable):
        assert len(data) == 2, "target size must be tuple of (w, h)"
        return tuple(data)
    else:
        raise ValueError("target size must be int or tuple of (w, h)") 
