# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import importlib
import random
import numpy as np
from typing import Mapping, Optional

import torch

def initiate_from_config(config: Mapping):
    assert "target" in config, f"Expected key `target` to initialize!"
    module, cls = config["target"].rsplit(".", 1)
    meta_class = getattr(importlib.import_module(module, package=None), cls)
    return meta_class(**config.get("params", dict()))

def initiate_from_config_recursively(config: Mapping):
    assert "target" in config, f"Expected key `target` to initialize!"
    update_config = {"target": config["target"], "params": {}}
    for k, v in config["params"].items():
        if isinstance(v, Mapping) and "target" in v:
            sub_instance = initiate_from_config_recursively(v)
            update_config["params"][k] = sub_instance
        else:
            update_config["params"][k] = v
    return initiate_from_config(update_config)

def seed_everything(seed: Optional[int] = None):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True