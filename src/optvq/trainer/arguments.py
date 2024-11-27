# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # arguments with high priority
    parser.add_argument("--seed", type=int, default=42,
                        help="The random seed.")
    parser.add_argument("--gpu", type=int, nargs="+", default=None,
                        help="The GPU ids to use.")
    parser.add_argument("--is_distributed", action="store_true", default=False)
    parser.add_argument("--config", type=str, default=None,
                        help="The path to the configuration file.")
    parser.add_argument("--resume", type=str, default=None,
                        help="The path to the checkpoint to resume.")
    parser.add_argument("--device_rank", type=int, default=0)

    # arguments for the training
    parser.add_argument("--log_dir", type=str, default=None,
                        help="The path to the log directory.")
    parser.add_argument("--mode", type=str, default="train",
                        help="options: train, test")
    parser.add_argument("--use_initiate", type=str, default=None, 
                        help="Options: random, kmeans")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--enterpoint", type=str, default=None)
    parser.add_argument("--code_path", type=str, default=None)
    parser.add_argument("--embed_path", type=str, default=None)
    
    # arguments for the data
    parser.add_argument("--use_train_subset", type=float, default=None,
                        help="The size of the training subset. None means using the full training set.")
    parser.add_argument("--use_train_repeat", type=int, default=None,
                        help="The number of times to repeat the training set.")
    parser.add_argument("--use_val_subset", type=float, default=None,
                        help="The size of the validation subset. None means using the full validation set.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulate", type=int, default=None)

    # arguments for the optimizer
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--mul_lr", type=float, default=None)

    # arguments for the model
    parser.add_argument("--num_codes", type=int, default=None,
                        help="The number of codes.")
    
    return parser