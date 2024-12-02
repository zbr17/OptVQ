# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

# Convert a Pytorch model to a Hugging Face model

import torch.nn as nn

from huggingface_hub import PyTorchModelHubMixin

from optvq.models.backbone.diffusion import Encoder, Decoder
from optvq.models.quantizer import VectorQuantizer, VectorQuantizerSinkhorn
from optvq.losses.aeloss_disc import AELossWithDisc
from optvq.models.vqgan import VQModel

class VQModelHF(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
            encoder: dict = {},
            decoder: dict = {},
            loss: dict = {},
            quantize: dict = {},
            quantize_type: str = "optvq",
            ckpt_path: str = None,
            ignore_keys=[],
            image_key="image",
            colorize_nlabels=None,
            monitor=None,
            use_connector: bool = True,
        ):
        super(VQModelHF, self).__init__()
        encoder = Encoder(**encoder)
        decoder = Decoder(**decoder)
        quantizer = self.setup_quantizer(quantize, quantize_type)
        loss = AELossWithDisc(**loss)

        self.model = VQModel(
            encoder=encoder,
            decoder=decoder,
            loss=loss,
            quantize=quantizer,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
            use_connector=use_connector,
        )
    
    def setup_quantizer(self, quantizer_config, quantize_type):
        if quantize_type == "optvq":
            quantizer = VectorQuantizerSinkhorn(**quantizer_config)
        elif quantize_type == "basevq":
            quantizer = VectorQuantizer(**quantizer_config)
        else:
            raise ValueError(f"Unknown quantizer type: {quantize_type}")
        return quantizer
    
    def encode(self, x):
        return self.model.encode(x)

    def decode(self, x):
        return self.model.decode(x)

    def forward(self, x):
        quant, *_ = self.encode(x)
        rec = self.decode(quant)
        return quant, rec