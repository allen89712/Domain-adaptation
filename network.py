import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
"""透過kernal調整feature map的維度"""
import torch
import torch.nn as nn



class rbgGenarator(nn.Module):
    def __init__(self, num_features, hidden_dim, h, w):
        super(rbgGenarator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, hidden_dim, kernel_size=1),  # 1x1 convolution
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),    # 1x1 convolution
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),    # 1x1 convolution
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_features*2, kernel_size=1),  # 1x1 convolution, output 2x the channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 最後加入AdaptiveAvgPool2d，將空間維度壓縮到 1x1
        )
    
    def forward(self, x):

        x = self.conv(x)
        
        gamma, beta = torch.split(x, x.size(1) // 2, dim=1)
        # print(f"gamma shape: {gamma.shape}, beta shape: {beta.shape}")
        return gamma, beta


# class ChannelWiseDynamicNorm(nn.Module):
#     def __init__(self, num_features, hidden_dim=64, eps=1e-5, momentum=0.1,h=1,w=1):
#         super(ChannelWiseDynamicNorm, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.rbgen = rbgGenarator(num_features, num_features, h, w)
#         self.register_buffer("running_mean", torch.zeros(num_features))
#         self.register_buffer("running_var", torch.ones(num_features))

#     def forward(self, x):
#         B, C, H, W = x.size()

#         if self.training:
#             mean = x.mean(dim=[0, 2, 3], keepdim=True)
#             var = x.var(dim=[0, 2, 3], keepdim=True, unbiased=False)

#             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.view(-1)
#             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.view(-1)

#             x_hat = (x - mean) / torch.sqrt(var + self.eps)
            
#         else:
#             mean = self.running_mean.view(1, C, 1, 1)
#             var = self.running_var.view(1, C, 1, 1)
#             x_hat = (x - mean) / torch.sqrt(var + self.eps)


#         gamma, beta = self.rbgen(x)

#         return gamma * x_hat + beta
    
class ChannelWiseDynamicNorm(nn.Module):
    def __init__(self, num_features, hidden_dim=64, eps=1e-5, momentum=0.1,h=1,w=1):
        super(ChannelWiseDynamicNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.rbgen = rbgGenarator(num_features, num_features, h, w)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        B, C, H, W = x.size()

        # if self.training:
        #     mean = x.mean(dim=[0, 2, 3], keepdim=True)
        #     var = x.var(dim=[0, 2, 3], keepdim=True, unbiased=False)

        #     self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.view(-1)
        #     self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.view(-1)

        #     x_hat = (x - mean) / torch.sqrt(var + self.eps)
            
        # else:
        #     mean = self.running_mean.view(1, C, 1, 1)
        #     var = self.running_var.view(1, C, 1, 1)
        #     x_hat = (x - mean) / torch.sqrt(var + self.eps)

        
        gamma, beta = self.rbgen(x)
        new_g = 0.9 + (1 - self.momentum) * gamma
        new_b = 0 + (1 - self.momentum) * beta

        return new_g * x + new_b
    
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=64),
            # ChannelWiseDynamicNorm(64,h=16,w=16),  # Use custom dynamic norm layer
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=128),
            # ChannelWiseDynamicNorm(128,h=8,w=8),  # Use custom dynamic norm layer
            nn.Dropout2d(0.3),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=256),
            # ChannelWiseDynamicNorm(256,h=4,w=4),  # Use custom dynamic norm layer
            nn.Dropout2d(0.5),
            nn.ReLU()
        ])  
        self.in_features = 256*4*4

    def forward(self, x):
        # 遍歷 conv_params 層，逐層處理
        for layer in self.conv_params:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return x

class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x