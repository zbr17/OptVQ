# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import torch.nn as nn

class PlainCNNEncoder(nn.Module):
    def __init__(self, in_dim: int = 3):
        super(PlainCNNEncoder, self).__init__()

        self.in_dim = in_dim

        self.in_fc = nn.Conv2d(in_channels=in_dim, out_channels=16,
                                kernel_size=3, stride=1, padding=1, bias=True)
        self.act0 = nn.ReLU(inplace=True)

        self.down1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, 
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.act1 = nn.ReLU(inplace=True)

        self.down2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.act2 = nn.ReLU(inplace=True)

        self.out_fc = nn.Conv2d(in_channels=32, out_channels=32,
                                kernel_size=3, stride=1, padding=1, bias=True)
    
    @property
    def hidden_dim(self):
        return 32

    def forward(self, x):
        x = self.in_fc(x)
        x = self.act0(x)

        x = self.down1(x)
        x = self.conv1(x)
        x = self.act1(x)

        x = self.down2(x)
        x = self.conv2(x)
        x = self.act2(x)

        x = self.out_fc(x)
        return x

class PlainCNNDecoder(nn.Module):
    def __init__(self, out_dim: int = 3):
        super(PlainCNNDecoder, self).__init__()
        self.out_dim = out_dim

        self.in_fc = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=1, padding=1, bias=True)
        
        self.act1 = nn.ReLU(inplace=True)
        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, 
                               kernel_size=3, stride=1, padding=1, bias=True)
        
        self.act2 = nn.ReLU(inplace=True)
        self.up2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, 
                               kernel_size=3, stride=1, padding=1, bias=True)
        
        self.act3 = nn.ReLU(inplace=True)
        self.out_fc = nn.Conv2d(in_channels=16, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1, bias=True)
    
    @property
    def hidden_dim(self):
        return 32
    
    def forward(self, x):
        x = self.in_fc(x)

        x = self.act1(x)
        x = self.up1(x)
        x = self.conv1(x)

        x = self.act2(x)
        x = self.up2(x)
        x = self.conv2(x)
        
        x = self.act3(x)
        x = self.out_fc(x)
        return x