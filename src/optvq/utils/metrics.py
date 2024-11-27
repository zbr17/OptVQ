# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import numpy as np

import torch

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

class FIDMetric:
    def __init__(self, device, dims=2048):
        self.device = device
        self.num_workers = 32
        
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)
        self.model.eval()

        self.reset_metrics()
    
    def reset_metrics(self):
        self.x_pred = []
        self.x_rec_pred = []

    @torch.no_grad()
    def get_activates(self, x: torch.Tensor):
        pred = self.model(x)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.squeeze().cpu().numpy()

    def update(self, x: torch.Tensor, x_rec: torch.Tensor):
        """
        Args:
            x (torch.Tensor): input tensor range from 0 to 1
            x_rec (torch.Tensor): reconstructed tensor range from 0 to 1
        """
        self.x_pred.append(self.get_activates(x))
        self.x_rec_pred.append(self.get_activates(x_rec))
    
    def result(self):
        assert len(self.x_pred) != 0, "No data to compute FID"
        x = np.concatenate(self.x_pred, axis=0)
        x_rec = np.concatenate(self.x_rec_pred, axis=0)

        x_mu = np.mean(x, axis=0)
        x_sigma = np.cov(x, rowvar=False)
        
        x_rec_mu = np.mean(x_rec, axis=0)
        x_rec_sigma = np.cov(x_rec, rowvar=False)

        fid_score = calculate_frechet_distance(x_mu, x_sigma, x_rec_mu, x_rec_sigma)
        self.reset_metrics()
        return fid_score