# Preventing Local Pitfalls in Vector Quantization via Optimal Transport 

Paper 
| [Project Page](https://boruizhang.site/OptVQ/) 
| [HF Demo](https://huggingface.co/spaces/BorelTHU/OptVQ)
| [中文解读](https://zhuanlan.zhihu.com/p/12811862624)

***Struggling with 'index collapse' in vector quantization? Discover OptVQ, a solution designed to maximize codebook utilization and enhance reconstruction quality.***

![head](assets/head.png)

## News

| [2024-12-16] We release the training code of OptVQ.  
| [2024-11-26] We release the pre-trained models of OptVQ.

## Introduction

We conduct image reconstruction experiments on the ImageNet dataset, and the quantitative comparison is shown below:

| Model | Latent Size | #Tokens | From Scratch | SSIM↑ | PSNR ↑ | LPIPS↓ | rFID↓ |
| - | - | - | - | - | - | - | - |
| taming-VQGAN | 16 × 16 | 1,024 | √ | 0.521 | 23.30 | 0.195 | 6.25 |
| MaskGiT-VQGAN | 16 × 16 | 1,024 | √ | - | - | - | 2.28 |
| Mo-VQGAN | 16 × 16 × 4 | 1,024 | √ | 0.673 | 22.42 | 0.113 | 1.12 |
| TiTok-S-128 | 128 | 4,096 | × | - | - | - | 1.71 |
| ViT-VQGAN | 32 × 32 | 8,192 | √ | - | - | - | 1.28 |
| taming-VQGAN | 16 × 16 | 16,384 | √ | 0.542 | 19.93 | 0.177 | 3.64 |
| RQ-VAE | 8 × 8 × 16 | 16,384 | √ | - | - | - | 1.83 |
| VQGAN-LC | 16 × 16 | 100,000 | × | 0.589 | 23.80 | 0.120 | 2.62 |
| OptVQ (ours) | 16 × 16 × 4 | 16,384 | √ | 0.717 | 26.59 | 0.076 | 1.00 |
| OptVQ (ours) | 16 × 16 × 8 | 16,384 | √ | 0.729 | 27.57 | 0.066 | 0.91 |

### Toy Example

We visualize the process of OptVQ and Vanilla VQ on a two-dimensional toy example.
The left figure with red points represents the baseline (Vanilla VQ), and the right figure with green points represents the proposed method (OptVQ).
<p float="left">
  <img src="assets/base.gif" width="300" />
  <img src="assets/sink.gif" width="300" />
</p>

## Installation

Please install the dependencies by running the following command:
```bash
# install the dependencies
pip install -r requirements.txt
# install the faiss-gpu package via conda
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
# install the optvq package
pip install -e .
```

## Usage: Quantizer

If you want to use our quantizer in your project, you can follow the code below:
```python
# Given the input tensor x, the quantizer will output the quantized tensor x_quant, the loss, and the indices.
from optvq.models.quantizer import VectorQuantizerSinkhorn
quantizer = VectorQuantizerSinkhorn(n_e=1024, e_dim=256, num_head=1)
x_quant, loss, indices = quantizer(x)
```

## Usage: VQ-VAE

### Inference

Please download the pre-trained models from the following links:

| Model | Link (Tsinghua) | Link (Hugging Face) |
| - | - | - |
| OptVQ (16 x 16 x 4) | [Download](https://cloud.tsinghua.edu.cn/d/91befd96f06a4a83bb03/) | [Download](https://huggingface.co/BorelTHU/optvq-16x16x4) |
| OptVQ (16 x 16 x 8) | [Download](https://cloud.tsinghua.edu.cn/d/309a55529e1f4f42a8d2/) | [Download](https://huggingface.co/BorelTHU/optvq-16x16x8) |

#### Option 1: Load from Hugging Face

You can load from the Hugging Face model hub by running the following code:
```python
# Example: load the OptVQ with 16 x 16 x 4
from optvq.models.vqgan_hf import VQModelHF
model = VQModelHF.from_pretrained("BorelTHU/optvq-16x16x4")
```

#### Option 2: Load from the local checkpoint

You can also write the following code to load the pre-trained model locally:
```python
# Example: load the OptVQ with 16 x 16 x 4
from optvq.utils.init import initiate_from_config_recursively
from omegaconf import OmegaConf
import torch
config = OmegaConf.load("configs/optvq.yaml")
model = initiate_from_config_recursively(config.autoencoder)
params = torch.load(..., map_location="cpu")
model.load_state_dict(params["model"])
```

#### Perform inference

After loading the model, you can perform inference (reconstruction):

```python
# load the dataset
dataset = ... # the input should be normalized to [-1, 1]
data = dataset[...] # size: (BS, C, H, W)

# reconstruct the input
with torch.no_grad():
    quant, *_ = model.encode(data)
    recon = model.decode(quant)
```

### Evaluation

To evaluate the model, you can use the following code:
```bash
config_path=configs/imagenet/optvq_256_f16_h4.yaml
log_dir=<path-to-log-folder>
resume=<path-to-checkpoint-folder>

python eval.py --config $config_path --log_dir $log_dir --resume $resume --is_distributed
```

### Training

We train the OptVQ (16 × 16 × 4) model on the ImageNet dataset with 8 NVIDIA 4090 GPUs for 50 epochs (around 8 days).
The training script is as follows:
```bash
config_path=configs/imagenet/optvq_256_f16_h4.yaml
log_dir=<path-to-log-folder>

python train.py --config $config_path --log_dir $log_dir --is_distributed --lr 2e-6
```

## Future work

- **Image Generation:** Due to limited computation resources, the generation experiment on ImageNet with MaskGiT will cost around more than 1 month on 8 NVIDIA 4090 GPUs. If you are interested in this experiment, please contact us.
- **High Compression Ratio:** We also plan to explore how to train a tokenizer with high compression ratio (e.g., f=32). If you have any ideas, please feel free to contact us. 

<!-- ## Citation

If you find this work useful, please consider citing it.

```bibtex
xxx
``` -->