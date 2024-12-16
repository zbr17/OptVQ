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

def get_setup_optimizers(config):
    name = config.train.pipeline
    func_name = "setup_optimizers_" + name
    return globals()[func_name]

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
def _find_weight_decay_id(modules: list, params_ids: list, 
                          include_class: tuple = (nn.Linear, nn.Conv2d, 
                                                  nn.ConvTranspose2d,
                                                  nn.MultiheadAttention), 
                          include_name: list = ["weight"]):
    for mod in modules:
        for sub_mod in mod.modules():
            if isinstance(sub_mod, include_class):
                for name, param in sub_mod.named_parameters():
                    if any([k in name for k in include_name]):
                        params_ids.append(id(param))
    params_ids = list(set(params_ids))
    return params_ids

def set_weight_decay(modules: list):
    weight_decay_ids = _find_weight_decay_id(modules, [])
    wd_params, wd_names, no_wd_params, no_wd_names = [], [], [], []
    for mod in modules:
        for name, param in mod.named_parameters():
            if id(param) in weight_decay_ids:
                wd_params.append(param)
                wd_names.append(name)
            else:
                no_wd_params.append(param)
                no_wd_names.append(name)
    return wd_params, wd_names, no_wd_params, no_wd_names

def setup_optimizers_ae(config: OmegaConf, model: nn.Module, total_steps: int):
    L.log.info("\n\n### Setting up the optimizers and schedulers. ###")
    
    # compute the total batch size and the learning rate
    total_batch_size = config.data.batch_size * config.world_size * config.data.gradient_accumulate
    total_learning_rate = config.train.learning_rate * total_batch_size
    multipled_learning_rate = total_learning_rate * config.train.mul_learning_rate
    L.log.info(f"Total batch size: {total_batch_size} = {config.data.batch_size} * {config.world_size} * {config.data.gradient_accumulate}")
    L.log.info(f"Total learning rate: {total_learning_rate} = {config.train.learning_rate} * {total_batch_size}")
    L.log.info(f"Multipled learning rate: {multipled_learning_rate} = {total_learning_rate} * {config.train.mul_learning_rate}")

    # setup the optimizers
    param_group = []
    ## base learning rate
    wd_params, wd_names, no_wd_params, no_wd_names = set_weight_decay([model.encoder, model.decoder, model.quant_conv, model.post_quant_conv])
    param_group.append({
        "params": wd_params, "lr": total_learning_rate, "eps": 1e-7,
        "weight_decay": config.train.weight_decay, "beta": (0.9, 0.999),
    })
    param_group.append({
        "params": no_wd_params, "lr": total_learning_rate, "eps": 1e-7,
        "weight_decay": 0.0, "beta": (0.9, 0.999),
    })
    ## multipled learning rate
    wd_params, wd_names, no_wd_params, no_wd_names = set_weight_decay([model.quantize])
    param_group.append({
        "params": wd_params, "lr": multipled_learning_rate, "eps": 1e-7,
        "weight_decay": config.train.weight_decay, "beta": (0.9, 0.999),
    })
    param_group.append({
        "params": no_wd_params, "lr": multipled_learning_rate, "eps": 1e-7,
        "weight_decay": 0.0, "beta": (0.9, 0.999),
    })

    optimizer_ae = torch.optim.AdamW(param_group)
    optimizer_dict = {"optimizer_ae": optimizer_ae}

    # setup the schedulers
    scheduler_ae = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer_ae, max_lr=[total_learning_rate, total_learning_rate, multipled_learning_rate, multipled_learning_rate],
        total_steps=total_steps, pct_start=0.01, anneal_strategy="cos"
    )
    scheduler_dict = {"scheduler_ae": scheduler_ae}

    # setup the scalers
    scaler_dict = {"scaler_ae": torch.cuda.amp.GradScaler(enabled=config.use_amp)}
    L.log.info(f"Enable AMP: {config.use_amp}")
    return optimizer_dict, scheduler_dict, scaler_dict

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

def setup_optimizers_ae_disc(config: OmegaConf, model: nn.Module, total_steps: int):
    L.log.info("\n\n### Setting up the optimizers and schedulers. ###")
    
    # compute the total batch size and the learning rate
    total_batch_size = config.data.batch_size * config.world_size * config.data.gradient_accumulate
    total_learning_rate = config.train.learning_rate * total_batch_size
    multipled_learning_rate = total_learning_rate * config.train.mul_learning_rate
    L.log.info(f"Total batch size: {total_batch_size} = {config.data.batch_size} * {config.world_size} * {config.data.gradient_accumulate}")
    L.log.info(f"Total learning rate: {total_learning_rate} = {config.train.learning_rate} * {total_batch_size}")
    L.log.info(f"Multipled learning rate: {multipled_learning_rate} = {total_learning_rate} * {config.train.mul_learning_rate}")

    # setup the optimizers
    param_group = []
    ## base learning rate
    wd_params, wd_names, no_wd_params, no_wd_names = set_weight_decay([model.encoder, model.decoder, model.quant_conv, model.post_quant_conv])
    param_group.append({
        "params": wd_params, "lr": total_learning_rate, "eps": 1e-7,
        "weight_decay": config.train.weight_decay, "beta": (0.9, 0.999),
    })
    param_group.append({
        "params": no_wd_params, "lr": total_learning_rate, "eps": 1e-7,
        "weight_decay": 0.0, "beta": (0.9, 0.999),
    })
    ## multipled learning rate
    wd_params, wd_names, no_wd_params, no_wd_names = set_weight_decay([model.quantize])
    param_group.append({
        "params": wd_params, "lr": multipled_learning_rate, "eps": 1e-7,
        "weight_decay": config.train.weight_decay, "beta": (0.9, 0.999),
    })
    param_group.append({
        "params": no_wd_params, "lr": multipled_learning_rate, "eps": 1e-7,
        "weight_decay": 0.0, "beta": (0.9, 0.999),
    })
    optimizer_ae = torch.optim.AdamW(param_group)

    param_group = []
    wd_params, wd_names, no_wd_params, no_wd_names = set_weight_decay([model.loss.discriminator])
    param_group.append({
        "params": wd_params, "lr": total_learning_rate, "eps": 1e-7,
        "weight_decay": config.train.weight_decay, "beta": (0.9, 0.999),
    })
    param_group.append({
        "params": no_wd_params, "lr": total_learning_rate, "eps": 1e-7,
        "weight_decay": 0.0, "beta": (0.9, 0.999),
    })
    optimizer_disc = torch.optim.AdamW(param_group)
    optimizer_dict = {"optimizer_ae": optimizer_ae, "optimizer_disc": optimizer_disc}

    # setup the schedulers
    scheduler_ae = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer_ae, max_lr=[total_learning_rate, total_learning_rate, multipled_learning_rate, multipled_learning_rate],
        total_steps=total_steps, pct_start=0.01, anneal_strategy="cos"
    )
    scheduler_disc = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer_disc, max_lr=[total_learning_rate, total_learning_rate],
        total_steps=total_steps, pct_start=0.01, anneal_strategy="cos"
    )
    scheduler_dict = {"scheduler_ae": scheduler_ae, "scheduler_disc": scheduler_disc}

    # setup the scalers
    scaler_dict = {"scaler_ae": torch.cuda.amp.GradScaler(enabled=config.use_amp), 
                   "scaler_disc": torch.cuda.amp.GradScaler(enabled=config.use_amp)}
    L.log.info(f"Enable AMP: {config.use_amp}")
    return optimizer_dict, scheduler_dict, scaler_dict

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