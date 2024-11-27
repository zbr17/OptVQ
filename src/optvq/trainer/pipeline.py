# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from typing import Callable
import argparse
import os
from omegaconf import OmegaConf
from functools import partial
from torchinfo import summary

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from optvq.utils.init import initiate_from_config_recursively
from optvq.data.dataloader import maybe_get_subset
import optvq.utils.logger as L

def setup_config(opt: argparse.Namespace):
    L.log.info("\n\n### Setting up the configurations. ###")

    # load the config files
    config = OmegaConf.load(opt.config)

    # overwrite the certain arguments according to the config.args mapping
    for key, value in config.args_map.items():
        if hasattr(opt, key) and getattr(opt, key) is not None:
            msg = f"config.{value} = opt.{key}"
            L.log.info(f"Overwrite the config: {msg}")
            exec(msg)
    
    return config

def setup_dataloader(data, batch_size, is_distributed: bool = True, is_train: bool = True, num_workers: int = 8):
    if is_train:
        if is_distributed:
            # setup the sampler
            sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=True, drop_last=True)
            # setup the dataloader
            loader = DataLoader(
                dataset=data, batch_size=batch_size, num_workers=num_workers,
                drop_last=True, sampler=sampler, persistent_workers=True, pin_memory=True
            )
        else:
            # setup the dataloader
            loader = DataLoader(
                dataset=data, batch_size=batch_size, num_workers=num_workers,
                drop_last=True, shuffle=True, persistent_workers=True, pin_memory=True
            )
    else:
        if is_distributed:
            # setup the sampler
            sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=False, drop_last=False)
            # setup the dataloader
            loader = DataLoader(
                dataset=data, batch_size=batch_size, num_workers=num_workers,
                drop_last=False, sampler=sampler, persistent_workers=True, pin_memory=True
            )
        else:
            # setup the dataloader
            loader = DataLoader(
                dataset=data, batch_size=batch_size, num_workers=num_workers,
                drop_last=False, shuffle=False, persistent_workers=True, pin_memory=True
            )
    
    return loader

def setup_dataset(config: OmegaConf):
    L.log.info("\n\n### Setting up the datasets. ###")

    # setup the training dataset
    train_data = initiate_from_config_recursively(config.data.train)
    if config.data.use_train_subset is not None:
        train_data = maybe_get_subset(train_data, subset_size=config.data.use_train_subset, num_data_repeat=config.data.use_train_repeat)
    L.log.info(f"Training dataset size: {len(train_data)}")

    # setup the validation dataset
    val_data = initiate_from_config_recursively(config.data.val)
    if config.data.use_val_subset is not None:
        val_data = maybe_get_subset(val_data, subset_size=config.data.use_val_subset)
    L.log.info(f"Validation dataset size: {len(val_data)}")

    return train_data, val_data

def setup_model(config: OmegaConf, device):
    L.log.info("\n\n### Setting up the models. ###")

    # setup the model
    model = initiate_from_config_recursively(config.model.autoencoder)
    if config.is_distributed:
        # apply syncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # model to devices
        model = model.to(device)
        find_unused_parameters = True
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[config.gpu],
            find_unused_parameters=find_unused_parameters
        )
        model_ori = model.module
    else:
        model = model.to(device)
        model_ori = model
    
    input_size = config.data.train.params.transform.params.resize
    in_channels = getattr(model_ori.encoder, "in_dim", 3)
    sout = summary(model_ori, (1, in_channels, input_size, input_size), device="cuda", verbose=0)
    L.log.info(sout)

    # count the total number of parameters
    for name, module in model_ori.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        L.log.info(f"Module: {name}, Total params: {num_params}, Trainable params: {num_trainable}")

    return model

### factory functions

def get_pipeline(config):
    name = config.train.pipeline
    func_name = "pipeline_" + name
    return globals()[func_name]

def _forward_backward(
    config,
    x: torch.Tensor,
    forward: Callable,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, 
                        enabled=config.use_amp):
        # forward pass
        loss, *output = forward(x)
        loss_acc = loss / config.data.gradient_accumulate
    scaler.scale(loss_acc).backward()
    # gradient accumulate
    if L.log.total_steps % config.data.gradient_accumulate == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        optimizer.zero_grad()
        scaler.update()
    
    if scheduler is not None:
        scheduler.step()
    return loss, output

### autoencoder version

def pipeline_ae(
    config,
    x: torch.Tensor,
    model: nn.Module,
    optimizers: dict,
    schedulers: dict,
    scalers: dict,
):
    assert "optimizer_ae" in optimizers
    assert "scheduler_ae" in schedulers
    assert "scaler_ae" in scalers

    optimizer = optimizers["optimizer_ae"]
    scheduler = schedulers["scheduler_ae"]
    scaler = scalers["scaler_ae"]

    forward = partial(model, mode=0)
    _, (loss_ae_dict, indices) = _forward_backward(config, x, forward, model, optimizer, scheduler, scaler)
    
    log_per_step = loss_ae_dict
    log_per_epoch = {"indices": indices}
    return log_per_step, log_per_epoch

### autoencoder + disc version

def pipeline_ae_disc(
    config, 
    x: torch.Tensor,
    model: nn.Module,
    optimizers: dict,
    schedulers: dict,
    scalers: dict,
):
    # autoencoder step
    assert "optimizer_ae" in optimizers
    assert "scheduler_ae" in schedulers
    assert "scaler_ae" in scalers

    optimizer = optimizers["optimizer_ae"]
    scheduler = schedulers["scheduler_ae"]
    scaler = scalers["scaler_ae"]

    forward = partial(model, mode=0)
    _, (loss_ae_dict, indices) = _forward_backward(config, x, forward, model, optimizer, scheduler, scaler)
    
    log_per_step = loss_ae_dict
    log_per_epoch = {"indices": indices}

    # discriminator step
    assert "optimizer_disc" in optimizers
    assert "scheduler_disc" in schedulers
    assert "scaler_disc" in scalers

    optimizer = optimizers["optimizer_disc"]
    scheduler = schedulers["scheduler_disc"]
    scaler = scalers["scaler_disc"]

    forward = partial(model, mode=1)
    _, (loss_disc_dict, _) = _forward_backward(config, x, forward, model, optimizer, scheduler, scaler)
    log_per_step.update(loss_disc_dict)
    return log_per_step, log_per_epoch