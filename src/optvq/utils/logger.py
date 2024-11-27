# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import time
import datetime
from typing import List
import functools
import os
from PIL import Image
from termcolor import colored
import sys
import logging
from omegaconf import OmegaConf
import json

try:
    from torch.utils.tensorboard import SummaryWriter
    from torch import Tensor
    import torch
except:
    raise ImportError("Please install torch to use this module!")

"""
NOTE: The `log` instance is a global variable, which should be imported by other modules as:
    `import optvq.utils.logger as logger` 
rather than 
    `from optvq.utils.logger import log`.
"""

def setup_printer(file_log_dir: str, use_console: bool = True):
    printer = logging.getLogger("LOG")
    printer.setLevel(logging.DEBUG)
    printer.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
    
    # create the console handler
    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        printer.addHandler(console_handler)

    # create the file handler
    file_handler = logging.FileHandler(os.path.join(file_log_dir, "record.txt"), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    )
    printer.addHandler(file_handler)

    return printer

@functools.lru_cache()
def config_loggers(log_dir: str, local_rank: int = 0, master_rank: int = 0):
    global log

    if local_rank == master_rank:
        log = LogManager(log_dir=log_dir, main_logger=True)
    else:
        log = LogManager(log_dir=log_dir, main_logger=False)

class ProgressWithIndices:
    def __init__(self, total: int, sep_char: str = "| ", 
                 num_per_row: int = 4):
        self.total = total
        self.sep_char = sep_char
        self.num_per_row = num_per_row

        self.count = 0
        self.start_time = time.time()
        self.past_time = None
        self.current_time = None
        self.eta = None
        self.speed = None
        self.used_time = 0

    def update(self):
        self.count += 1
        if self.count <= self.total:
            self.past_time = self.current_time
            self.current_time = time.time()
            # compute eta
            if self.past_time is not None:
                self.eta = (self.total - self.count) * (self.current_time - self.past_time)
                self.eta = str(datetime.timedelta(seconds=int(self.eta)))
                self.speed = 1 / (self.current_time - self.past_time + 1e-8)
            # compute used time
            self.used_time = self.current_time - self.start_time
            self.used_time = str(datetime.timedelta(seconds=int(self.used_time)))
        else:
            self.eta = 0
            self.speed = 0
            self.past_time = None
            self.current_time = None

    def print(self, prefix: str = "", content: str = "", ):
        global log
        prefix_str = f"{prefix}\t" + f"[{self.count}/{self.total} {self.used_time}/Eta:{self.eta}], Speed:{self.speed}iters/s\n"
        content_list = content.split(self.sep_char)
        content_list = [content.strip() for content in content_list]
        content_list = [
            "\t\t" + self.sep_char.join(content_list[i:i + self.num_per_row]) 
            for i in range(0, len(content_list), self.num_per_row)
        ]
        content = prefix_str + "\n".join(content_list)
        log.info(content)

class LogManager:
    """
    This class encapsulates the tensorboard writer, the statistic meters, the console printer, and the progress counters.

    Args:
        log_dir (str): the parent directory to save all the logs
        init_meters (List[str]): the initial meters to be shown
        show_avg (bool): whether to show the average value of the meters
    """
    def __init__(self, log_dir: str, init_meters: List[str] = [], 
                 show_avg: bool = True, main_logger: bool = False):
        
        # initiate all the directories
        self.show_avg = show_avg
        self.log_dir = log_dir
        self.main_logger = main_logger
        self.setup_dirs()

        # initiate the statistic meters
        self.meters = {meter: AverageMeter() for meter in init_meters}
        
        # initiate the progress counters
        self.total_steps = 0
        self.total_epochs = 0

        if self.main_logger:
            # initiate the tensorboard writer
            self.board = SummaryWriter(log_dir=self.tb_log_dir)

            # initiate the console printer
            self.printer = setup_printer(self.file_log_dir, use_console=True)
    
    def state_dict(self):
        return {
            "total_steps": self.total_steps,
            "total_epochs": self.total_epochs,
            "meters": {
                meter_name: meter.state_dict() for meter_name, meter in self.meters.items()
            }
        }
    
    def load_state_dict(self, state_dict: dict):
        self.total_steps = state_dict["total_steps"]
        self.total_epochs = state_dict["total_epochs"]
        for meter_name, meter_state_dict in state_dict["meters"].items():
            if meter_name not in self.meters:
                self.meters[meter_name] = AverageMeter()    
            self.meters[meter_name].load_state_dict(meter_state_dict)
    
    ### About directories
    def setup_dirs(self):
        """
        The structure of the log directory:
        - log_dir: [tb_log, txt_log, img_log, model_log]
        """
        self.tb_log_dir = os.path.join(self.log_dir, "tb_log")
        # NOTE: For now, we save the txt records in the parent directory
        # self.file_log_dir = os.path.join(self.log_dir, "txt_log")
        self.file_log_dir = self.log_dir 
        self.img_log_dir = os.path.join(self.log_dir, "img_log")

        self.config_path = os.path.join(self.log_dir, "config.yaml")
        self.checkpoint_path = os.path.join(self.log_dir, "checkpoint.pth")
        self.backup_checkpoint_path = os.path.join(self.log_dir, "checkpoint.pth")
        self.save_logger_path = os.path.join(self.log_dir, "logger.json")

        if self.main_logger:
            os.makedirs(self.tb_log_dir, exist_ok=True)
            os.makedirs(self.file_log_dir, exist_ok=True)
            os.makedirs(self.img_log_dir, exist_ok=True)

    ### About printer

    def info(self, msg, *args, **kwargs):
        if self.main_logger:
            self.printer.info(msg, *args, **kwargs)

    def show(self, include_key: str = ""):
        if isinstance(include_key, str):
            include_key = [include_key]
        if self.show_avg:
            return "| ".join([f"{meter_name}: {meter.val:.4f}/{meter.avg:.4f}" for meter_name, meter in self.meters.items() if any([k in meter_name for k in include_key])])
        else:
            return "| ".join([f"{meter_name}: {meter.val:.4f}" for meter_name, meter in self.meters.items() if any([k in meter_name for k in include_key])])

    ### About counter

    def update_steps(self):
        self.total_steps += 1
        return self.total_steps
    
    def update_epochs(self):
        self.total_epochs += 1
        return self.total_epochs
    
    ### About tensorboard
    def add_histogram(self, tag: str, values: Tensor, global_step: int = None):
        if self.main_logger:
            global_step = self.total_steps if global_step is None else global_step
            self.board.add_histogram(tag, values, global_step)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = None):
        if isinstance(scalar_value, Tensor):
            scalar_value = scalar_value.item()
        if tag in self.meters:
            cur_step = self.meters[tag].update(scalar_value)
            cur_step = cur_step if global_step is None else global_step
            if self.main_logger:
                self.board.add_scalar(tag, scalar_value, cur_step)
        else:
            self.meters[tag] = AverageMeter()
            cur_step = self.meters[tag].update(scalar_value)
            cur_step = cur_step if global_step is None else global_step
            if self.main_logger:
                print(f"Create new meter: {tag}!")
                self.board.add_scalar(tag, scalar_value, cur_step)
    
    def add_scalar_dict(self, scalar_dict: dict, global_step: int = None):
        for tag, scalar_value in scalar_dict.items():
            self.add_scalar(tag, scalar_value, global_step)
        
    def add_images(self, tag: str, images: Tensor, global_step: int = None):
        if self.main_logger:
            global_step = self.total_steps if global_step is None else global_step
            self.board.add_images(tag, images, global_step, dataformats="NCHW")

    ### About saving and resuming
    def save_configs(self, config):
        if self.main_logger:
            # save config as yaml file
            OmegaConf.save(config, self.config_path)
            self.info(f"Save config to {self.config_path}.")
            
            # save logger
            state_dict = self.state_dict()
            with open(self.save_logger_path, "w") as f:
                json.dump(state_dict, f)
    
    def load_configs(self):
        # load config
        assert os.path.exists(self.config_path), f"Config {self.config_path} does not exist!"
        config = OmegaConf.load(self.config_path)

        # load logger
        assert os.path.exists(self.save_logger_path), f"Logger {self.save_logger_path} does not exist!"
        state_dict = json.load(open(self.save_logger_path, "r"))
        self.load_state_dict(state_dict)

        return config

    def save_checkpoint(self, model, optimizers, schedulers, scalers, suffix: str = ""):
        """
        checkpoint_dict: model, optimizer, scheduler, scalers
        """
        if self.main_logger:
            
            # save checkpoint_dict
            checkpoint_dict = {
                "model": model.state_dict(),
                "epoch": self.total_epochs,
                "step": self.total_steps
            }
            checkpoint_dict.update({k: v.state_dict() for k, v in optimizers.items()})
            checkpoint_dict.update({k: v.state_dict() for k, v in schedulers.items() if v is not None})
            checkpoint_dict.update({k: v.state_dict() for k, v in scalers.items()})

            checkpoint_path = self.checkpoint_path + suffix
            torch.save(checkpoint_dict, checkpoint_path)
            if os.path.exists(self.backup_checkpoint_path):
                os.remove(self.backup_checkpoint_path)
            self.backup_checkpoint_path = checkpoint_path + f".epoch{self.total_epochs}"
            torch.save(checkpoint_dict, self.backup_checkpoint_path)

            self.info(f"### Epoch: {self.total_epochs}| Steps: {self.total_steps}| Save checkpoint to {checkpoint_path}.")
    
    def load_checkpoint(self, device, model, optimizers, schedulers, scalers, resume: str = None):
        resume_path = self.checkpoint_path if resume is None else resume
        assert os.path.exists(resume_path), f"Resume {resume_path} does not exist!"

        # load checkpoint_dict
        checkpoint_dict = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint_dict["model"])
        self.total_epochs = checkpoint_dict["epoch"]
        self.total_steps = checkpoint_dict["step"]
        for k, v in optimizers.items():
            v.load_state_dict(checkpoint_dict[k])
        for k, v in schedulers.items():
            v.load_state_dict(checkpoint_dict[k])
        for k, v in scalers.items():
            v.load_state_dict(checkpoint_dict[k])
        
        self.info(f"### Epoch: {self.total_epochs}| Steps: {self.total_steps}| Resume checkpoint from {resume_path}.")

        return self.total_epochs

class EmptyManager:
    def __init__(self):
        for func_name in LogManager.__dict__.keys():
            if not func_name.startswith("_"):
                setattr(self, func_name, lambda *args, **kwargs: print(f"Empty Manager! {func_name} is not available!"))

class AverageMeter:
    def __init__(self):
        self.reset()

    def state_dict(self):
        return {
            "val": self.val,
            "avg": self.avg,
            "sum": self.sum,
            "count": self.count,
        }
    
    def load_state_dict(self, state_dict: dict):
        self.val = state_dict["val"]
        self.avg = state_dict["avg"]
        self.sum = state_dict["sum"]
        self.count = state_dict["count"]

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self.count

    def __str__(self):
        return f"{self.avg:.4f}"

def save_image(x: Tensor, save_path: str, scale_to_256: bool = True):
    """
    Args:
        x (tensor): default data range is [0, 1]
    """
    if scale_to_256:
        x = x.mul(255).clamp(0, 255)
    x = x.permute(1, 2, 0).detach().cpu().numpy().astype("uint8")
    img = Image.fromarray(x)
    img.save(save_path)

def save_images(images_list, ids_list, meta_path):
    for i, (image, id) in enumerate(zip(images_list, ids_list)):
        save_path = os.path.join(meta_path, f"{id}.png")
        save_image(image, save_path)

def save_images_multithread(images_list, ids_list, meta_path):
    n_workers = 32
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for i in range(0, len(images_list), n_workers):
            cur_images = images_list[i:(i + n_workers)]
            cur_ids = ids_list[i:(i + n_workers)]
            executor.submit(save_images, cur_images, cur_ids, meta_path)

def add_prefix(log_dict: dict, prefix: str):
    return {
        f"{prefix}/{key}": val for key, val in log_dict.items()
    }

##################### GLOBAL VARIABLES #####################
log = EmptyManager()
GET_STATS: bool = (os.environ.get("ENABLE_STATS", "1") == "1")
###########################################################