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

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

from optvq.models.discriminator import NLayerDiscriminator, weights_init

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class AELossWithDisc(nn.Module):
    def __init__(self, 
                 disc_start, 
                 pixelloss_weight=1.0,
                 disc_in_channels=3, 
                 disc_num_layers=3, 
                 use_actnorm=False, 
                 disc_ndf=64, 
                 disc_conditional=False,
                 disc_loss="hinge",
                 loss_l1_weight: float = 1.0,
                 loss_l2_weight: float = 1.0,
                 loss_p_weight: float = 1.0,
                 loss_q_weight: float = 1.0,
                 loss_g_weight: float = 1.0,
                 loss_d_weight: float = 1.0
        ):
        super(AELossWithDisc, self).__init__()
        assert disc_loss in ["hinge", "vanilla"]
        
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = lpips.LPIPS(net="vgg", verbose=False).eval()

        self.loss_l1_weight = loss_l1_weight
        self.loss_l2_weight = loss_l2_weight
        self.loss_p_weight = loss_p_weight
        self.loss_q_weight = loss_q_weight
        self.loss_g_weight = loss_g_weight
        self.loss_d_weight = loss_d_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")

        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        g_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        g_weight = torch.clamp(g_weight, 0.0, 1e4).detach()
        g_weight = g_weight * self.loss_g_weight
        
        # detection nan
        if torch.isnan(g_weight).any():
            g_weight = torch.tensor(0.0, device=g_weight.device)
        return g_weight

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, codebook_loss, inputs, reconstructions, mode, last_layer=None, cond=None, global_step=0):
        x = inputs.contiguous().float()
        x_rec = reconstructions.contiguous().float()

        # compute q loss
        loss_q = codebook_loss.mean()

        # compute l1 loss
        loss_l1 = (x_rec - x).abs().mean() if self.loss_l1_weight > 0.0 else torch.tensor(0.0, device=x.device)

        # compute l2 loss
        loss_l2 = (x_rec - x).pow(2).mean() if self.loss_l2_weight > 0.0 else torch.tensor(0.0, device=x.device)

        # compute perceptual loss
        loss_p = self.perceptual_loss(x, x_rec).mean() if self.loss_p_weight > 0.0 else torch.tensor(0.0, device=x.device)

        # intigrate reconstruction loss
        loss_rec = loss_l1 * self.loss_l1_weight + \
                   loss_l2 * self.loss_l2_weight + \
                   loss_p * self.loss_p_weight

        # setup the factor_disc
        if global_step < self.discriminator_iter_start:
            factor_disc = 0.0
        else:
            factor_disc = 1.0

        # now the GAN part
        if mode == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(x_rec)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((x_rec, cond), dim=1))

            # compute g loss
            loss_g = - logits_fake.mean()

            try:
                loss_g_weight = self.calculate_adaptive_weight(loss_rec, loss_g, last_layer=last_layer)
            except RuntimeError:
                # assert not self.training
                loss_g_weight = torch.tensor(0.0)

            loss = loss_g * loss_g_weight * factor_disc  + \
                   loss_q * self.loss_q_weight + \
                   loss_rec

            log = {"total_loss": loss.item(),
                   "loss_q": loss_q.item(),
                   "loss_rec": loss_rec.item(),
                   "loss_l1": loss_l1.item(),
                   "loss_l2": loss_l2.item(),
                   "loss_p": loss_p.item(),
                   "loss_g": loss_g.item(),
                   "loss_g_weight": loss_g_weight.item(),
                   "factor_disc": factor_disc,
            }
            return loss, log

        if mode == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(x.detach())
                logits_fake = self.discriminator(x_rec.detach())
            else:
                logits_real = self.discriminator(torch.cat((x.detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((x_rec.detach(), cond), dim=1))

            loss_d = self.disc_loss(logits_real, logits_fake).mean()
            loss = loss_d * self.loss_d_weight

            log = {"loss_d": loss_d.item(),
                   "logits_real": logits_real.mean().item(),
                   "logits_fake": logits_fake.mean().item()
            }
            return loss, log