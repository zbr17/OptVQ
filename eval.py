# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from omegaconf import OmegaConf
import os
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import pyiqa

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp

from optvq.trainer.pipeline import (
    setup_dataset, setup_config,
    setup_dataloader, setup_model
)
from optvq.trainer.arguments import get_parser
import optvq.utils.logger as L
from optvq.utils.init import seed_everything
from optvq.data.preprocessor import get_recover_map
from optvq.utils.func import dist_all_gather
from optvq.utils.metrics import FIDMetric

_USE_TORCHRUN_: bool = False

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

def main_worker(gpu, ngpus_per_node, opt):
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    # setup the loggers and configs
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
    
    # setup the datasets
    train_data, val_data = setup_dataset(config)
    train_loader = setup_dataloader(train_data, config.data.batch_size, config.is_distributed, is_train=True)
    val_loader = setup_dataloader(val_data, config.data.test_batch_size, config.is_distributed, is_train=False)

    # setup the models
    model = setup_model(config, device)
    model_ori = model.module if config.is_distributed else model

    # resume from the checkpoint
    resume_path = os.path.join(opt.resume, "checkpoint.pth")
    assert os.path.exists(resume_path), f"Resume {resume_path} does not exist!"
    checkpoint_dict = torch.load(resume_path, map_location=device)
    model_ori.load_state_dict(checkpoint_dict["model"])
    
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