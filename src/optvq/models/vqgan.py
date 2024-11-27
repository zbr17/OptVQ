# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------
# Modified from [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import torch.nn as nn

import optvq.utils.logger as L

class Identity(nn.Module):
    def forward(self, x):
        return x

class VQModel(nn.Module):
    def __init__(self,
            encoder: nn.Module,
            decoder: nn.Module,
            loss: nn.Module,
            quantize: nn.Module,
            ckpt_path: str = None,
            ignore_keys=[],
            image_key="image",
            colorize_nlabels=None,
            monitor=None,
            use_connector: bool = True,
        ):
        super(VQModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.quantize = quantize
        self.use_connector = use_connector

        encoder_dim = self.encoder.hidden_dim
        decoder_dim = self.decoder.hidden_dim
        embed_dim = self.quantize.e_dim
        
        if not use_connector:
            self.quant_conv = Identity()
            self.post_quant_conv = Identity()
            assert encoder_dim == embed_dim, f"{encoder_dim} != {embed_dim}"
            assert decoder_dim == embed_dim, f"{decoder_dim} != {embed_dim}"
        else:
            self.quant_conv = torch.nn.Conv2d(encoder_dim, embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder_dim, 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, indices = self.quantize(h)
        return quant, emb_loss, indices

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x, mode: int = 0, global_step: int = None):
        """
        Args:
            x (torch.Tensor): input tensor
            mode (int): 0 for autoencoder, 1 for discriminator
            global_step (int): global step for adaptive discriminator weight
        """
        global_step = global_step if global_step is not None else L.log.total_steps
        quant, qloss, indices = self.encode(x)
        xrec = self.decode(quant)
        if mode == 0:
            # compute the autoencoder loss
            loss, log_dict = self.loss(qloss, x, xrec, mode, last_layer=self.get_last_layer(), global_step=global_step)
        elif mode == 1:
            # compute the discriminator loss
            loss, log_dict = self.loss(qloss, x, xrec, mode, last_layer=self.get_last_layer(), global_step=global_step)
        elif mode == 2:
            # compute the hidden embedding
            h = self.encoder(x)
            h = self.quant_conv(h)
            return h
        return loss, log_dict, indices

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def get_last_layer(self):
        if hasattr(self.decoder, "conv_out"):
            return self.decoder.conv_out.weight
        elif hasattr(self.decoder, "out_fc"):
            return self.decoder.out_fc.weight
        elif hasattr(self.decoder, "inv_conv"):
            return self.decoder.inv_conv.weight
        else:
            raise NotImplementedError(f"Cannot find last layer in decoder")

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    # The functions below are deprecated

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict