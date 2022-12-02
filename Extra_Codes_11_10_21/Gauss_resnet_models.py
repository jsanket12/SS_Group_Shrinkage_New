import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from Gauss_layers_new import SSGauss_Node_layer, SSGauss_Edge_layer, Gauss_layer
from Gauss_Conv_layers_new import SSGauss_Node_Conv2d_layer, SSGauss_Edge_Conv2d_layer, Gauss_Conv2d_layer
from Gauss_BN_layers import SSGauss_VB_BatchNorm2d, Gauss_VB_BatchNorm2d

import numpy as np

__all__ = ['resnet']

##########################################################################################################################################
#### Spike-and-slab node selection with Gaussian: Resnet and Wide-Resnet
def SSGauss_Node_conv3x3(in_planes, out_planes, stride=1, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
    "3x3 convolution with padding"
    return SSGauss_Node_Conv2d_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)

class SSGauss_Node_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()
        self.conv1 = SSGauss_Node_conv3x3(inplanes, planes, stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn1 = SSGauss_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv2 = SSGauss_Node_conv3x3(planes, planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn2 = SSGauss_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        # self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x))) # self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.silu(out) # self.relu(out) 

        return out

class SSGauss_Node_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()
        self.conv1 = SSGauss_Node_Conv2d_layer(inplanes, planes, kernel_size=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn1 = SSGauss_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv2 = SSGauss_Node_Conv2d_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn2 = SSGauss_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv3 = SSGauss_Node_Conv2d_layer(planes, planes * 4, kernel_size=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn3 = SSGauss_VB_BatchNorm2d(planes * 4, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = F.silu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.silu(out)

        return out

class SSGauss_Node_ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = SSGauss_Node_Bottleneck if depth >=44 else SSGauss_Node_BasicBlock

        self.inplanes = 16
        self.sigma_0 = sigma_0
        self.temp = temp
        self.gamma_prior = gamma_prior
        self.conv1 = SSGauss_Node_Conv2d_layer(3, 16, kernel_size=3, padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn1 = SSGauss_VB_BatchNorm2d(16, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        # self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = Gauss_layer(64 * block.expansion, num_classes, sigma_0=sigma_0)

        # for m in self.modules():
        #     if isinstance(m, SSGauss_Node_Conv2d_layer):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.w_mu.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, SSGauss_VB_BatchNorm2d):
        #         m.weight_mu.data.fill_(1)
        #         m.bias_mu.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SSGauss_Node_Conv2d_layer(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0),
                SSGauss_VB_BatchNorm2d(planes * block.expansion, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x))) # self.relu(self.bn1(self.conv1(x)))    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SSGauss_Node_Wide_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()
        self.bn1 = SSGauss_VB_BatchNorm2d(inplanes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv1 = SSGauss_Node_conv3x3(inplanes, planes, stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn2 = SSGauss_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv2 = SSGauss_Node_conv3x3(planes, planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)        

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                SSGauss_Node_Conv2d_layer(inplanes, planes, kernel_size=1, stride=stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0),
            )

    def forward(self, x):
        out = self.conv1(F.silu(self.bn1(x)))
        out = self.conv2(F.silu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class SSGauss_Node_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=1000, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.in_planes = 16
        self.sigma_0 = sigma_0
        self.temp = temp
        self.gamma_prior = gamma_prior
        self.conv1 = SSGauss_Node_conv3x3(3,nStages[0], temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.layer1 = self._wide_layer(SSGauss_Node_Wide_BasicBlock, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(SSGauss_Node_Wide_BasicBlock, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(SSGauss_Node_Wide_BasicBlock, nStages[3], n, stride=2)
        self.bn1 = SSGauss_VB_BatchNorm2d(nStages[3], momentum=0.9, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = Gauss_layer(nStages[3], num_classes, sigma_0=sigma_0)
        self.prune_flag = 0
        self.mask = None

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.silu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

##########################################################################################################################################
#### Spike-and-slab edge selection with Gaussian: Resnet and Wide-Resnet
def SSGauss_Edge_conv3x3(in_planes, out_planes, stride=1, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
    "3x3 convolution with padding"
    return SSGauss_Edge_Conv2d_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)

class SSGauss_Edge_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()
        self.conv1 = SSGauss_Edge_conv3x3(inplanes, planes, stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn1 = SSGauss_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv2 = SSGauss_Edge_conv3x3(planes, planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn2 = SSGauss_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        # self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x))) # self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.silu(out) # self.relu(out)

        return out

class SSGauss_Edge_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()
        self.conv1 = SSGauss_Edge_Conv2d_layer(inplanes, planes, kernel_size=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn1 = SSGauss_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv2 = SSGauss_Edge_Conv2d_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn2 = SSGauss_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv3 = SSGauss_Edge_Conv2d_layer(planes, planes * 4, kernel_size=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn3 = SSGauss_VB_BatchNorm2d(planes * 4, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = F.silu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.silu(out)

        return out

class SSGauss_Edge_ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = SSGauss_Edge_Bottleneck if depth >=44 else SSGauss_Edge_BasicBlock

        self.inplanes = 16
        self.sigma_0 = sigma_0
        self.temp = temp
        self.gamma_prior = gamma_prior
        self.conv1 = SSGauss_Edge_Conv2d_layer(3, 16, kernel_size=3, padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn1 = SSGauss_VB_BatchNorm2d(16, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        # self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = SSGauss_Edge_layer(64 * block.expansion, num_classes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)

        # for m in self.modules():
        #     if isinstance(m, SSGauss_Edge_Conv2d_layer):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.w_mu.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, SSGauss_Edge_layer):
        #         m.weight_mu.data.fill_(1)
        #         m.bias_mu.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SSGauss_Edge_Conv2d_layer(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0),
                SSGauss_VB_BatchNorm2d(planes * block.expansion, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x))) # self.relu(self.bn1(self.conv1(x)))    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SSGauss_Edge_Wide_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()
        self.bn1 = SSGauss_VB_BatchNorm2d(inplanes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv1 = SSGauss_Edge_conv3x3(inplanes, planes, stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn2 = SSGauss_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv2 = SSGauss_Edge_conv3x3(planes, planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)        

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                SSGauss_Edge_Conv2d_layer(inplanes, planes, kernel_size=1, stride=stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0),
            )

    def forward(self, x):
        out = self.conv1(F.silu(self.bn1(x)))
        out = self.conv2(F.silu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class SSGauss_Edge_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.in_planes = 16
        self.sigma_0 = sigma_0
        self.temp = temp
        self.gamma_prior = gamma_prior
        self.conv1 = SSGauss_Edge_conv3x3(3,nStages[0], temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.layer1 = self._wide_layer(SSGauss_Edge_Wide_BasicBlock, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(SSGauss_Edge_Wide_BasicBlock, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(SSGauss_Edge_Wide_BasicBlock, nStages[3], n, stride=2)
        self.bn1 = SSGauss_VB_BatchNorm2d(nStages[3], momentum=0.9, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = SSGauss_Edge_layer(nStages[3], num_classes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)  

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.silu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

##########################################################################################################################################
#### Gaussian without spike-and-slab: Resnet and Wide-Resnet
def Gauss_conv3x3(in_planes, out_planes, stride=1, sigma_0 = 1):
    "3x3 convolution with padding"
    return Gauss_Conv2d_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, sigma_0=sigma_0)

class Gauss_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, sigma_0 = 1):
        super().__init__()
        self.conv1 = Gauss_conv3x3(inplanes, planes, stride, sigma_0=sigma_0)
        self.bn1 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
        self.conv2 = Gauss_conv3x3(planes, planes, sigma_0=sigma_0)
        self.bn2 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
        # self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x))) # self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.silu(out) # self.relu(out)

        return out

class Gauss_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, sigma_0 = 1):
        super().__init__()
        self.conv1 = Gauss_Conv2d_layer(inplanes, planes, kernel_size=1, bias=False, sigma_0=sigma_0)
        self.bn1 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
        self.conv2 = Gauss_Conv2d_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, sigma_0=sigma_0)
        self.bn2 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
        self.conv3 = Gauss_Conv2d_layer(planes, planes * 4, kernel_size=1, bias=False, sigma_0=sigma_0)
        self.bn3 = Gauss_VB_BatchNorm2d(planes * 4, sigma_0=sigma_0)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = F.silu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.silu(out)

        return out

class Gauss_ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, sigma_0 = 1):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Gauss_Bottleneck if depth >=44 else Gauss_BasicBlock

        self.inplanes = 16
        self.sigma_0 = sigma_0
        self.conv1 = Gauss_Conv2d_layer(3, 16, kernel_size=3, padding=1, bias=False, sigma_0=sigma_0)
        self.bn1 = Gauss_VB_BatchNorm2d(16, sigma_0=sigma_0)
        # self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = Gauss_layer(64 * block.expansion, num_classes, sigma_0=sigma_0)

        # for m in self.modules():
        #     if isinstance(m, Gauss_Conv2d_layer):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.w_mu.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, Gauss_VB_BatchNorm2d):
        #         m.weight_mu.data.fill_(1)
        #         m.bias_mu.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Gauss_Conv2d_layer(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, sigma_0=self.sigma_0),
                Gauss_VB_BatchNorm2d(planes * block.expansion,sigma_0=self.sigma_0),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, sigma_0=self.sigma_0))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, sigma_0=self.sigma_0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x))) # self.relu(self.bn1(self.conv1(x)))    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Gauss_Wide_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, sigma_0 = 1):
        super().__init__()
        self.bn1 = Gauss_VB_BatchNorm2d(inplanes, sigma_0=sigma_0)
        self.conv1 = Gauss_conv3x3(inplanes, planes, stride, sigma_0=sigma_0)
        self.bn2 = Gauss_VB_BatchNorm2d(planes, sigma_0=sigma_0)
        self.conv2 = Gauss_conv3x3(planes, planes, sigma_0=sigma_0)        

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                Gauss_Conv2d_layer(inplanes, planes, kernel_size=1, stride=stride, sigma_0=sigma_0),
            )

    def forward(self, x):
        out = self.conv1(F.silu(self.bn1(x)))
        out = self.conv2(F.silu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Gauss_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, sigma_0 = 1):
        super().__init__()

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.in_planes = 16
        self.sigma_0 = sigma_0
        self.conv1 = Gauss_conv3x3(3,nStages[0], sigma_0=sigma_0)
        self.layer1 = self._wide_layer(Gauss_Wide_BasicBlock, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(Gauss_Wide_BasicBlock, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(Gauss_Wide_BasicBlock, nStages[3], n, stride=2)
        self.bn1 = Gauss_VB_BatchNorm2d(nStages[3], momentum=0.9, sigma_0=sigma_0)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = Gauss_layer(nStages[3], num_classes, sigma_0=sigma_0)

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sigma_0=self.sigma_0))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.silu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out