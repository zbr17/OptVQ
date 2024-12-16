# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from omegaconf import OmegaConf
import os
import time
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import pyiqa
from collections import defaultdict

try:
    from faiss import Kmeans
except:
    warnings.warn("Faiss is not installed. The kmeans clustering will not be available.")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp

from optvq.trainer.pipeline import (
    get_pipeline, get_setup_optimizers, 
    setup_config, setup_dataset, 
    setup_dataloader, setup_model
)
from optvq.trainer.arguments import get_parser
import optvq.utils.logger as L
from optvq.utils.init import seed_everything
from optvq.data.preprocessor import get_recover_map
from optvq.utils.func import dist_all_gather
from optvq.utils.metrics import FIDMetric

_USE_TORCHRUN_: bool = False

def generate_embeds(config: OmegaConf, device: torch.device, model: nn.Module, data):
    model.eval()
    model_ori = model.module if config.is_distributed else model
    N = model_ori.quantize.n_e
    D = model_ori.quantize.e_dim
    data_path = os.path.join(L.log.log_dir, "embed.pth")

    L.log.info(f"Start sampling the data.")
    loader = setup_dataloader(data, batch_size=config.data.gen_embed_batch_size, is_distributed=config.is_distributed, is_train=True)
    loader.sampler.set_epoch(0)
    downsample_ratio = 4
    with torch.no_grad():
        embed_dict = defaultdict(list)
        pbar = tqdm(enumerate(loader), total=len(loader)) if config.local_rank == 0 else enumerate(loader)
        for i, (x, label) in pbar:
            x = x.to(device)
            label = label.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=config.use_amp):
                h = model(x, mode=2)
            if h.ndim == 4:
                ks_size = h.size(-1) // 4
                h = torch.nn.functional.avg_pool2d(h, kernel_size=ks_size, stride=ks_size)
                h = h.permute(0, 2, 3, 1).contiguous().view(h.size(0), -1, D)
            else:
                pos_ids = torch.randperm(h.size(1))[:8]
                h = h[:, pos_ids]
            HW = h.size(1)

            num_sample = int(h.size(0) / downsample_ratio)
            sample_ids = torch.randperm(h.size(0))[:num_sample]
            h = h[sample_ids]
            label = label[sample_ids]
            if config.is_distributed:
                h = dist_all_gather(h)
                label = dist_all_gather(label)

            for X, ids in zip(h, label):
                embed_dict[ids.item()].append(X)
            if config.local_rank == 0:                    
                pbar.set_description(f"#C = {len([_ for _ in embed_dict.values() if len(_) > 0])} #P = {sum([len(_) * HW for _ in embed_dict.values()])}")
            
            if config.is_distributed:
                dist.barrier()
        
    # combine the embeddings
    for k in list(embed_dict.keys()):
        embed_dict[k] = torch.cat(embed_dict[k], dim=0).view(-1, D)
    if config.local_rank == 0:
        torch.save(embed_dict, data_path)
        L.log.info(f"Save the embeddings to {data_path}")
    
    if config.is_distributed:
        dist.barrier()
    print(f"RANK reached: {config.local_rank}")
    L.log.info(f"Embed size: {sum([_.size(0) for _ in embed_dict.values()])}")

def generate_codes(config: OmegaConf, device: torch.device):
    assert not config.is_distributed, "The codebook generation must be non-distributed."

    method = config.train.use_initiate
    N = config.model.autoencoder.params.quantize.params.n_e
    D = config.model.autoencoder.params.quantize.params.e_dim
    num_head = config.model.autoencoder.params.quantize.params.num_head

    data_path = os.path.join(L.log.log_dir, "embed.pth") if config.train.embed_path is None else config.train.embed_path
    code_path = os.path.join(L.log.log_dir, "codebook.pth")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Embedding file not found at {data_path}")
    embed_dict = torch.load(data_path, map_location=device)

    if method == "kmeans":
        L.log.info(f"Initiate with {method} method.")
        # run the kmeans clustering

        L.log.info(f"Start the kmeans clustering.")
        num_classes = len([_ for _ in embed_dict.values() if len(_) > 0])
        num_clusters_per_class = N // num_classes + 1
        code_list = []
        pbar = tqdm(enumerate(embed_dict.items()), total=len(embed_dict))
        for i, (k, v) in pbar:
            if len(v) == 0:
                continue
            kmeans = Kmeans(d=D, k=num_clusters_per_class, verbose=False, niter=100, gpu=True)
            kmeans.train(v.float().cpu().numpy())
            sub_codes = torch.from_numpy(kmeans.centroids).to(device)
            code_list.append(sub_codes)

        codes = torch.cat(code_list, dim=0)
        assert codes.size(0) >= N, f"Code size is {codes.size(0)} while the target size is {N}"
        if codes.size(0) > N:
            sampled_ids = torch.randperm(codes.size(0))[:N]
            codes = codes[sampled_ids]        
        # save the embeddings
        torch.save(codes, code_path)
    elif method == "random":
        L.log.info(f"Initiate with {method} method.")
        embed_to_select = [v for v in embed_dict.values() if len(v) > 0]
        embed_to_select = torch.cat(embed_to_select, dim=0)
        embed_to_select = embed_to_select.reshape(-1, int(D / num_head))
        sampled_ids = torch.randperm(embed_to_select.size(0))[:N]
        codes = embed_to_select[sampled_ids]
        # save the embeddings
        torch.save(codes, code_path)

def train_one_epoch(config: OmegaConf, device: torch.device, 
                    model: nn.Module, loader: DataLoader,
                    optimizers: dict, schedulers: dict, scalers: dict):
    # set the model to train mode
    model.train()
    model_ori = model.module if config.is_distributed else model

    # codebook utility meter
    num_code = model_ori.quantize.n_e
    codebook_usage = torch.zeros(num_code, device=device)

    # iterate over the dataloader
    pipeline_func = get_pipeline(config)
    iterator = enumerate(loader)
    pbar = L.ProgressWithIndices(total=len(loader))
    for i, (x, _) in iterator:
        pbar.update()
        L.log.update_steps()
        x = x.to(device)

        log_per_step, log_per_epoch = pipeline_func(
            config=config, x=x, model=model, optimizers=optimizers, 
            schedulers=schedulers, scalers=scalers,
        )

        # log per step
        log_per_step = L.add_prefix(log_per_step, "train")
        L.log.add_scalar_dict(log_per_step)

        # log the indices
        if "indices" in log_per_epoch:
            indices = log_per_epoch["indices"]
            if indices is not None:
                indices = indices.view(-1)
                codebook_usage.index_add_(0, indices, torch.ones_like(indices, dtype=codebook_usage.dtype))

        if i % 20 == 0 and config.local_rank == 0:
            pbar.print(
                prefix=f"Epoch: {L.log.total_epochs} / Iters: {L.log.total_steps}-TRAIN",
                content=L.log.show(["train", "params"])
            )
        
        # visualize the reconstruction results
        if L.log.total_steps % config.train.visualize_interval == 0:
            
            x, x_rec = visualize_batch(config, x, model_ori)
            L.log.add_images("train/rec", x_rec)
            L.log.add_images("train/x", x)
            
        # log the learning rate
        for name, optimizer in optimizers.items():
            for idx, param_group in enumerate(optimizer.param_groups):
                L.log.add_scalar(f"LR/{name}_{idx}", param_group["lr"])
    
    # log the codebook usage each epoch
    codebook_usage = torch.sum(codebook_usage > 0).item() / len(codebook_usage)
    L.log.add_scalar("train/codebook_usage", codebook_usage)

def evaluate(config: OmegaConf, device: torch.device, model: nn.Module, loader: DataLoader):
    # set the model to eval
    model.eval()
    model_ori = model.module if config.is_distributed else model
    recover_map = get_recover_map(config.data.preprocess)

    # codebook utility meter
    num_code = model_ori.quantize.n_e
    codebook_usage = torch.zeros(num_code, device=device)

    # meters
    lpips_list = []
    psnr_list = []
    ssim_list = []

    psnr_computer = pyiqa.create_metric(
        metric_name="psnr", test_y_channel=True, data_range=1.0, color_space="ycbcr", device=device
    )
    ssim_computer = pyiqa.create_metric(
        metric_name="ssim", device=device
    )
    lpips_computer = pyiqa.create_metric(
        metric_name="lpips", color_space="ycbcr", device=device
    )
    fid_computer = FIDMetric(device=device)

    # iterate over the dataloader
    iterator = tqdm(enumerate(loader), total=len(loader))
    pbar = L.ProgressWithIndices(total=len(loader))
    with torch.no_grad():
        for i, (x, _) in iterator:
            pbar.update()
            L.log.update_steps()
            x = x.to(device)

            quant, _, indices = model_ori.encode(x)
            x_rec = model_ori.decode(quant)

            x = recover_map(x).float()
            x_rec = recover_map(x_rec).float()
            x_rec = x_rec.clamp(min=0.0, max=1.0)

            # visualize the reconstruction results
            if i == 0:
                max_vis_num = 8
                L.log.add_images("val/rec", x_rec[:max_vis_num])
                L.log.add_images("val/x", x[:max_vis_num])

            if config.is_distributed:
                # gather the data
                x = dist_all_gather(x)
                x_rec = dist_all_gather(x_rec)
            
            lpips_score = lpips_computer(x, x_rec)
            lpips_list.append(lpips_score)

            psnr_score = psnr_computer(x, x_rec)
            psnr_list.append(psnr_score)

            ssim_score = ssim_computer(x, x_rec)
            ssim_list.append(ssim_score)

            fid_computer.update(x, x_rec)

            if indices is not None:
                indices = indices.view(-1)
                codebook_usage.index_add_(0, indices, torch.ones_like(indices, dtype=codebook_usage.dtype))
            
            if config.is_distributed:
                dist.barrier()
        
        if config.local_rank == 0:
            psnr_score = torch.cat(psnr_list).mean().item()
            ssim_score = torch.cat(ssim_list).mean().item()
            lpips_score = torch.cat(lpips_list).mean().item()
            fid_score = fid_computer.result()
            L.log.add_scalar("val/psnr", psnr_score)
            L.log.add_scalar("val/ssim", ssim_score)
            L.log.add_scalar("val/lpips", lpips_score)
            L.log.add_scalar("val/fid", fid_score)
            pbar.print(
                prefix=f"Evaluation: ",
                content=L.log.show("val")
            )

@torch.no_grad()
def visualize_batch(config: OmegaConf, x: torch.Tensor, model: nn.Module):
    """
    Visualize the reconstruction results
    """
    max_vis_num = 8
    recover_map = get_recover_map(config.data.preprocess)

    model.eval()
    quant, _, _ = model.encode(x)
    x_rec = model.decode(quant)
    x = recover_map(x).float()
    x_rec = recover_map(x_rec).float()
    model.train()

    return x[:max_vis_num], x_rec[:max_vis_num]

def main_worker(gpu, ngpus_per_node, opt):
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    # setup the loggers and configs
    if opt.resume is not None:
        L.config_loggers(log_dir=opt.resume, local_rank=gpu)
        L.log.info(f"Resume an existing training logger at {opt.resume}")
        config = L.log.load_configs()
        L.log.info(f"Load the configurations from {opt.resume}")
    else:
        L.config_loggers(log_dir=opt.log_dir, local_rank=gpu)
        L.log.info(f"Start a new training logger at {opt.log_dir}")
        config = setup_config(opt)
        L.log.save_configs(config)
        L.log.info(f"Save the configurations to {opt.log_dir}")
    
    # overwrite certain configurations
    config.is_distributed = opt.is_distributed
    config.resume = opt.resume
    config.mode = opt.mode
    config.gpu = gpu
    ## distributed configurations
    config.local_rank = int(gpu)
    config.world_size = int(ngpus_per_node * config.train.nnodes)
    config.world_rank = int(opt.device_rank * ngpus_per_node + gpu)

    # setup torch.distributed
    if config.is_distributed:
        if _USE_TORCHRUN_: 
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(
                backend="nccl", init_method=config.train.dist_url,
                rank=config.world_rank, world_size=config.world_size
            )
        dist.barrier()
    
    #####################################
    # Stage 1.b: generate the codebooks
    #####################################
    if config.mode == "gen_codes":
        # generate the codebook
        generate_codes(config=config, device=device)
        return
    
    # setup the datasets
    train_data, val_data = setup_dataset(config)
    train_loader = setup_dataloader(train_data, config.data.batch_size, config.is_distributed, is_train=True)
    val_loader = setup_dataloader(val_data, config.data.test_batch_size, config.is_distributed, is_train=False)

    # setup the models
    model = setup_model(config, device)
    model_ori = model.module if config.is_distributed else model

    #####################################
    # Stage 1.a: generate the codebooks
    #####################################
    if config.mode == "gen_embeds":
        # model initialization
        generate_embeds(config=config, device=device, model=model, data=train_data)
        return
    

    #####################################
    # Stage 2: model training or testing
    #####################################

    # setup the optimizers and schedulers
    total_steps = int(config.train.epochs * len(train_loader))
    optimizer_dict, scheduler_dict, scaler_dict = get_setup_optimizers(config)(config, model_ori, total_steps)

    # save configs or resume
    if config.resume is None:
        # save model
        L.log.save_checkpoint(model_ori, optimizer_dict, scheduler_dict, scaler_dict)
        start_epoch = 0
    else:
        # resume from the checkpoint
        start_epoch = L.log.load_checkpoint(device, model_ori, optimizer_dict, scheduler_dict, scaler_dict)
    
    # model enterpoint
    if config.train.enterpoint is not None:
        params = torch.load(config.train.enterpoint, map_location=device)["model"]
        # delete the quantize embeddings
        if params.get("quantize.embedding.weight", None) is not None:
            del params["quantize.embedding.weight"]
        elif params.get("_orig_mod.quantize.embedding.weight", None) is not None:
            del params["_orig_mod.quantize.embedding.weight"]
        missing_keys, unexpected_keys = model_ori.load_state_dict(params, strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
        L.log.info(f"Loaded the model from {config.train.enterpoint}")
        dist.barrier()

    if config.mode == "train":
        # load the codebooks
        code_path = os.path.join(L.log.log_dir, "codebook.pth") if config.train.code_path is None else config.train.code_path
        try:
            L.log.info(f"Load the codebook from {code_path}")
            codes = torch.load(code_path, map_location=device)
            model_ori.quantize.embedding.weight.data = codes
        except:
            L.log.info(f"Codebook not found at {code_path}. Use the random codebook.")
            codes = model_ori.quantize.embedding.weight.data
        print(f"Embedding {codes.size()} in rank {config.local_rank} = {codes[0][:5]}")
        
        # start training
        L.log.info("\n\n### Start training. ###")
        for epoch in range(start_epoch, config.train.epochs):
            L.log.update_epochs()

            if config.is_distributed:
                # set the seed for the epoch
                train_loader.sampler.set_epoch(epoch)

            # train one epoch
            try:
                train_one_epoch(
                    config=config, device=device, model=model, loader=train_loader, 
                    optimizers=optimizer_dict, schedulers=scheduler_dict, scalers=scaler_dict
                )
            except Exception as e:
                L.log.save_checkpoint(model_ori, optimizer_dict, scheduler_dict, scaler_dict, suffix=".error")
                raise e

            # evaluate the model
            evaluate(
                config=config, device=device, model=model, loader=val_loader
            )

            # save model
            L.log.save_checkpoint(model_ori, optimizer_dict, scheduler_dict, scaler_dict)
            L.log.save_configs(config)
    elif config.mode == "test":
        # start testing
        L.log.info("\n\n### Start testing. ###")
        evaluate(
            config=config, device=device, model=model, loader=val_loader
        )

def main():
    # parse the arguments
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # setup seed
    if opt.seed is not None:
        seed_everything(opt.seed)
        print(f"Seed is set to {opt.seed}")
    
    # setup devices
    if opt.gpu is None:
        opt.gpu = list(range(torch.cuda.device_count()))
    ngpus_per_node = len(opt.gpu)
    
    if opt.is_distributed:
        assert len(opt.gpu) > 1, "Expected more than 1 GPU for distributed training."
        if _USE_TORCHRUN_:
            main_worker(int(os.environ["LOCAL_RANK"]), ngpus_per_node, opt)
        else:
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        assert isinstance(opt.gpu, int) or len(opt.gpu) == 1, "Expected a single GPU for non-distributed training."
        opt.gpu = opt.gpu[0] if isinstance(opt.gpu, list) else opt.gpu
        main_worker(opt.gpu, ngpus_per_node, opt)

if __name__ == "__main__":
    main()