# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------
# Modified from [thuanz123/enhancing-transformers](https://github.com/thuanz123/enhancing-transformers)
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------

from typing import Tuple

import lpips

import torch
import torch.nn as nn

class AELoss(nn.Module):
    """
    Args:
        loss_q_weight (float): weight for quantization loss
        loss_l1_weight (float): weight for L1 loss (loglaplace)
        loss_l2_weight (float): weight for L2 loss (loggaussian)
        loss_p_weight (float): weight for perceptual loss
    """
    def __init__(self, loss_q_weight: float = 1.0,
                 loss_l1_weight: float = 1.0,
                 loss_l2_weight: float = 1.0,
                 loss_p_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_type = ["aeloss"]

        self.perceptual_loss = lpips.LPIPS(net="vgg", verbose=False)
        # freeze the perceptual loss
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

        self.loss_q_weight = loss_q_weight 
        self.loss_l1_weight = loss_l1_weight 
        self.loss_l2_weight = loss_l2_weight
        self.loss_p_weight = loss_p_weight 

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, q_loss: torch.FloatTensor, 
                x: torch.FloatTensor, 
                x_rec: torch.FloatTensor, *args, **kwargs) -> Tuple:
        x = x.float()
        x_rec = x_rec.float()
        
        # compute l1 loss
        loss_l1 = (x_rec - x).abs().mean() if self.loss_l1_weight > 0.0 else torch.tensor(0.0, device=x.device)

        # compute l2 loss
        loss_l2 = (x_rec - x).pow(2).mean() if self.loss_l2_weight > 0.0 else torch.tensor(0.0, device=x.device)

        # compute perceptual loss
        loss_p = self.perceptual_loss(x, x_rec).mean() if self.loss_p_weight > 0.0 else torch.tensor(0.0, device=x.device)

        # compute total loss
        loss = self.loss_p_weight * loss_p + \
               self.loss_l1_weight * loss_l1 + \
               self.loss_l2_weight * loss_l2
        loss += self.loss_q_weight * q_loss

        # get the log
        log = {
            "loss": loss.detach(),
            "loss_p": loss_p.detach(),
            "loss_l1": loss_l1.detach(),
            "loss_l2": loss_l2.detach(),
            "loss_q": q_loss.detach()
        }

        return loss, log