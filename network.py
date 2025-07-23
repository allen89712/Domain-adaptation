import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

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

        if self.training:
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            var = x.var(dim=[0, 2, 3], keepdim=True, unbiased=False)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.view(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.view(-1)

            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            
        else:
            mean = self.running_mean.view(1, C, 1, 1)
            var = self.running_var.view(1, C, 1, 1)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)


        gamma, beta = self.rbgen(x)

        return gamma * x_hat + beta
    
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
#         gamma, beta = self.rbgen(x)
#         new_g = 0.9 + (1 - self.momentum) * gamma
#         new_b = 0 + (1 - self.momentum) * beta

#         return new_g * x + new_b
    
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

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

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19, 
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = ChannelWiseDynamicNorm(num_features=64, h=56, w=56)  # 根據ResNet的結構，第一層BN的通道數是64
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        
        # 替換layer1中的BN層
        self.layer1 = self._replace_bn(model_resnet.layer1)
        self.layer2 = self._replace_bn(model_resnet.layer2)
        self.layer3 = self._replace_bn(model_resnet.layer3)
        self.layer4 = self._replace_bn(model_resnet.layer4)
        
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def _replace_bn(self, layer):
        """替換層中的所有BN層為ChannelWiseDynamicNorm"""
        for name, module in layer.named_children():
            if isinstance(module, nn.BatchNorm2d):
                # 創建新的ChannelWiseDynamicNorm層
                new_norm = ChannelWiseDynamicNorm(
                    num_features=module.num_features,
                    h=module.num_features,  # 這裡的h和w需要根據實際特徵圖大小調整
                    w=module.num_features
                )
                setattr(layer, name, new_norm)
            elif isinstance(module, nn.Sequential):
                # 遞歸處理Sequential中的層
                self._replace_bn(module)
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y