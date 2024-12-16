# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch.distributed as dist

import optvq.utils.logger as L

class VectorQuantizer(nn.Module):
    def __init__(self, n_e: int = 1024, e_dim: int = 128, 
                 beta: float = 1.0, use_norm: bool = False,
                 use_proj: bool = True, fix_codes: bool = False,
                 loss_q_type: str = "ce",
                 num_head: int = 1,
                 start_quantize_steps: int = None):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.loss_q_type = loss_q_type
        self.num_head = num_head
        self.start_quantize_steps = start_quantize_steps
        self.code_dim = self.e_dim // self.num_head

        self.norm = lambda x: F.normalize(x, p=2.0, dim=-1, eps=1e-6) if use_norm else x
        assert not use_norm, f"use_norm=True is no longer supported! Because the norm operation without theorectical analysis may cause unpredictable unstability."
        self.use_proj = use_proj

        self.embedding = nn.Embedding(num_embeddings=n_e, embedding_dim=self.code_dim)
        if use_proj:
            self.proj = nn.Linear(self.code_dim, self.code_dim)
            torch.nn.init.normal_(self.proj.weight, std=self.code_dim ** -0.5)
        if fix_codes:
            self.embedding.weight.requires_grad = False

    def reshape_input(self, x: Tensor):
        """
        (B, C, H, W) / (B, T, C) -> (B, T, C)
        """
        if x.ndim == 4:
            _, C, H, W = x.size()
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, H * W, C)
            return x, {"size": (H, W)}
        elif x.ndim == 3:
            return x, None
        else:
            raise ValueError("Invalid input shape!")

    def recover_output(self, x: Tensor, info):
        if info is not None:
            H, W = info["size"]
            if x.ndim == 3: # features (B, T, C) -> (B, C, H, W)
                C = x.size(2)
                return x.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
            elif x.ndim == 2: # indices (B, T) -> (B, H, W)
                return x.view(-1, H, W)
            else:
                raise ValueError("Invalid input shape!")
        else: # features (B, T, C) or indices (B, T)
            return x
    
    def get_codebook(self, return_numpy: bool = True):
        embed = self.proj(self.embedding.weight) if self.use_proj else self.embedding.weight
        if return_numpy:
            return embed.data.cpu().numpy()
        else:
            return embed.data

    def quantize_input(self, query, reference):
        # compute the distance matrix
        query2ref = torch.cdist(query, reference, p=2.0) # (B1, B2)

        # find the nearest embedding
        indices = torch.argmin(query2ref, dim=-1) # (B1,)
        nearest_ref = reference[indices] # (B1, C)
            
        return indices, nearest_ref, query2ref

    def compute_codebook_loss(self, query, indices, nearest_ref, beta: float, query2ref):
        # compute the loss
        if self.loss_q_type == "l2":
            loss = torch.mean((query - nearest_ref.detach()).pow(2)) + \
                   torch.mean((nearest_ref - query.detach()).pow(2)) * beta
        elif self.loss_q_type == "l1":
            loss = torch.mean((query - nearest_ref.detach()).abs()) + \
                   torch.mean((nearest_ref - query.detach()).abs()) * beta
        elif self.loss_q_type == "ce":
            loss = F.cross_entropy(- query2ref, indices)

        return loss
    
    def compute_quantized_output(self, x, x_q):
        if self.start_quantize_steps is not None:
            if self.training and L.log.total_steps < self.start_quantize_steps:
                L.log.add_scalar("params/quantize_ratio", 0.0)
                return x
            else:
                L.log.add_scalar("params/quantize_ratio", 1.0)
                return x + (x_q - x).detach()
        else:
            L.log.add_scalar("params/quantize_ratio", 1.0)
            return x + (x_q - x).detach()  

    def embed_code(self, code, size=None, code_format="image"):
        """
        Args:
            code_format (str): "image" (B x nH x H x W) or "sequence" (B x T x nH)
        """
        code = code.view(code.shape[0], -1)
        B, dim = code.size()
        if size is not None:
            H, W = size[0], size[1]
        else:
            H_W = int(dim / self.num_head)
            H = W = int(math.sqrt(H_W))
        assert H * W * self.num_head == dim

        embed = self.proj(self.embedding.weight) if self.use_proj else self.embedding.weight
        embed = self.norm(embed)
        x_q = embed[code] # (B, TxnH, dC)

        if code_format == "image":
            x_q = x_q.view(B, self.num_head, H_W, -1)
            x_q = x_q.permute(0, 2, 1, 3).contiguous().view(B, H_W, -1)
        elif code_format == "sequence":
            x_q = x_q.view(B, H_W, -1)
        x_q = self.recover_output(x_q, {"size": (H, W)})
        return x_q

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: Tensor):
        """
        Quantize the input tensor x with the embedding table.

        Args:
            x (Tensor): input tensor with shape (B, C, H, W) or (B, T, C)
        Returns:
            (tuple) containing: (x_q, loss, indices)
        """
        x = x.float()
        x, info = self.reshape_input(x)
        B, T, C = x.size()
        x = x.view(-1, C) # (B * T, C)
        embed = self.proj(self.embedding.weight) if self.use_proj else self.embedding.weight

        # split the x if multi-head is used
        if self.num_head > 1:
            x = x.view(-1, self.code_dim) # (B * T * nH, dC)

        # compute the distance between x and each embedding
        x, embed = self.norm(x), self.norm(embed)

        # compute losses
        indices, x_q, query2ref = self.quantize_input(x, embed)
        loss = self.compute_codebook_loss(
            query=x, indices=indices, nearest_ref=x_q, 
            beta=self.beta, query2ref=query2ref
        )

        # compute statistics
        if self.training and L.GET_STATS:
            with torch.no_grad():
                num_unique = torch.unique(indices).size(0)
                x_norm_mean = torch.mean(x.norm(dim=-1))
                embed_norm_mean = torch.mean(embed.norm(dim=-1))
                diff_norm_mean = torch.mean((x_q - x).norm(dim=-1))
                x2e_mean = query2ref.mean()
                L.log.add_scalar("params/num_unique", num_unique)
                L.log.add_scalar("params/x_norm", x_norm_mean.item())
                L.log.add_scalar("params/embed_norm", embed_norm_mean.item())
                L.log.add_scalar("params/diff_norm", diff_norm_mean.item())
                L.log.add_scalar("params/x2e_mean", x2e_mean.item())
    
        # compute the final x_q
        x_q = self.compute_quantized_output(x, x_q).view(B, T, C)
        indices = indices.view(B, T, self.num_head)

        # for output
        x_q = self.recover_output(x_q, info)
        indices = self.recover_output(indices, info)
        
        return x_q, loss, indices

def sinkhorn(cost: Tensor, n_iters: int = 3, epsilon: float = 1, is_distributed: bool = False):
    """
    Sinkhorn algorithm.
    Args:
        cost (Tensor): shape with (B, K)
    """
    Q = torch.exp(- cost * epsilon).t() # (K, B)
    if is_distributed:
        B = Q.size(1) * dist.get_world_size()
    else:
        B = Q.size(1)
    K = Q.size(0)

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if is_distributed:
        dist.all_reduce(sum_Q)
    Q /= (sum_Q + 1e-8)

    for _ in range(n_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if is_distributed:
            dist.all_reduce(sum_of_rows)
        Q /= (sum_of_rows + 1e-8)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
        Q /= B
    
    Q *= B # the columns must sum to 1 so that Q is an assignment
    return Q.t() # (B, K)

class VectorQuantizerSinkhorn(VectorQuantizer):
    def __init__(self, epsilon: float = 10.0, n_iters: int = 5, 
                 normalize_mode: str = "all", use_prob: bool = True,
                 *args, **kwargs):
        super(VectorQuantizerSinkhorn, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.normalize_mode = normalize_mode
        self.use_prob = use_prob

    def normalize(self, A, dim, mode="all"):
        if mode == "all":
            A = (A - A.mean()) / (A.std() + 1e-6)
            A = A - A.min()
        elif mode == "dim":
            A = A / math.sqrt(dim)
        elif mode == "null":
            pass
        return A

    def quantize_input(self, query, reference):
        # compute the distance matrix
        query2ref = torch.cdist(query, reference, p=2.0) # (B1, B2)
        
        # compute the assignment matrix
        with torch.no_grad():
            is_distributed = dist.is_initialized() and dist.get_world_size() > 1
            normalized_cost = self.normalize(query2ref, dim=reference.size(1), mode=self.normalize_mode)
            Q = sinkhorn(normalized_cost, n_iters=self.n_iters, epsilon=self.epsilon, is_distributed=is_distributed)
                
        if self.use_prob:
            # avoid the zero value problem
            max_q_id = torch.argmax(Q, dim=-1)
            Q[torch.arange(Q.size(0)), max_q_id] += 1e-8
            indices = torch.multinomial(Q, num_samples=1).squeeze()
        else:
            indices = torch.argmax(Q, dim=-1)
        nearest_ref = reference[indices]

        if self.training and L.GET_STATS:
            if L.log.total_steps % 1000 == 0:
                L.log.add_histogram("params/normalized_cost", normalized_cost)
        
        return indices, nearest_ref, query2ref

class Identity(VectorQuantizer):
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: Tensor):
        x = x.float()
        loss_q = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # compute statistics
        if self.training and L.GET_STATS:
            with torch.no_grad():
                x_flatten, _ = self.reshape_input(x)
                x_norm_mean = torch.mean(x_flatten.norm(dim=-1))
                L.log.add_scalar("params/x_norm", x_norm_mean.item())
        
        return x, loss_q, None