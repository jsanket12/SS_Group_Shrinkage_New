import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from Group_Lasso_linear_layers_non_center import Group_Lasso_layer
from Group_Lasso_Conv_layers_non_center import SS_Group_Lasso_Node_Conv2d_layer, Group_Lasso_Conv2d_layer
from Group_Lasso_BN_layers_non_center import SS_Group_Lasso_VB_BatchNorm2d, Group_Lasso_VB_BatchNorm2d
from rescale_bias import Bias2D

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

__all__ = ['resnet']

##########################################################################################################################################
#### Spike-and-slab node selection with Group Lasso: Resnet and Wide-Resnet
def SS_Group_Lasso_Node_conv3x3(in_planes, out_planes, lamb_mu, lamb_rho, stride=1, 
                        temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
    "3x3 convolution with padding"
    
    return SS_Group_Lasso_Node_Conv2d_layer(in_planes, out_planes, lamb_mu, lamb_rho, kernel_size=3, stride=stride,
                     padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)

class SS_Group_Lasso_Node_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, block_idx, max_block, lamb_mu, lamb_rho, stride=1, 
                temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__() 
        self.conv1 = SS_Group_Lasso_Node_conv3x3(inplanes, planes, lamb_mu, lamb_rho, stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = SS_Group_Lasso_Node_conv3x3(planes, planes, lamb_mu, lamb_rho, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.stride = stride

        self.addbias1 = Bias2D(inplanes)
        self.addbias2 = Bias2D(planes)

        self._scale = nn.Parameter(torch.ones(1))

        multiplier = (block_idx + 1) ** -(1/6) * max_block **(1/6)

        for m in self.modules():
            if isinstance(m, SS_Group_Lasso_Node_Conv2d_layer):
                _, C, H, W = m.w_mu.shape
                stddev = (C * H * W / 2) ** -.5
                nn.init.normal_(m.w_mu, std = stddev * multiplier)

        self.residual = max_block ** -.5
        self.identity = block_idx ** .5 / (block_idx + 1) ** .5
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            # avgpool = nn.AvgPool2d(stride) if stride != 1 else nn.Sequential()
            # self.shortcut = LambdaLayer(
            #     lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            self.shortcut = nn.Sequential(
                Bias2D(num_features=inplanes),
                SS_Group_Lasso_Node_Conv2d_layer(inplanes, planes, lamb_mu, lamb_rho, 
                                kernel_size=1, bias=False, stride=stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0))
            nn.init.kaiming_normal_(self.shortcut[1].w_mu, a=1)

    def forward(self, x):
        out = self.silu(self.conv1(self.addbias1(x)))
        out = self.conv2(self.addbias2(out))

        out = out.mul(self._scale.mul(self.residual))
        out = torch.add(input=out, alpha=self.identity, other=self.shortcut(x))
        
        out = self.silu(out)

        return out

class SS_Group_Lasso_Node_RescaleNet(nn.Module):
    def __init__(self, depth, num_classes=1000, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, lamb_c = 4, lamb_d = 2):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.block_idx = 3*n - 1
        self.max_depth = 3*n
        self.sigma_0 = sigma_0
        self.lamb_c = lamb_c
        self.lamb_d = lamb_d
        self.temp = temp
        self.gamma_prior = gamma_prior
        self.lamb_mu = nn.Parameter(torch.Tensor(1))        
        self.lamb_rho = nn.Parameter(torch.Tensor(1))      
        init.constant_(self.lamb_mu, 1)
        init.constant_(self.lamb_rho, -6.)

        self.kl_lambda = 0.

        block = SS_Group_Lasso_Node_BasicBlock

        self.inplanes = 16
        self.conv1 = SS_Group_Lasso_Node_Conv2d_layer(3, 16, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, 
                                    kernel_size=3, padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.addbias1 = Bias2D(self.inplanes)
        self.silu = nn.SiLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, stride=1)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.addbias2 = Bias2D(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = Group_Lasso_layer(64 * block.expansion, num_classes, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=sigma_0)

        nn.init.kaiming_normal_(self.conv1.w_mu)
        nn.init.kaiming_normal_(self.fc.w_mu, a=1)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.inplanes, planes, self.block_idx, self.max_depth, self.lamb_mu, self.lamb_rho, 
                            stride, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))
        
        self.inplanes = planes * block.expansion
        self.block_idx += 1
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.block_idx, self.max_depth, self.lamb_mu, self.lamb_rho,
                            temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))
            self.block_idx += 1

        return nn.Sequential(*layers)

    def forward(self, x):

        lamb_sigma = torch.log1p(torch.exp(self.lamb_rho))

        self.kl_lambda = -self.lamb_c*torch.log(self.lamb_d) + torch.lgamma(self.lamb_c) - \
                          self.lamb_c*self.lamb_mu + self.lamb_d*torch.exp(self.lamb_mu+(lamb_sigma**2/2)) - \
                          torch.log(lamb_sigma) - 1.41894
        
        x = self.silu(self.addbias1(self.conv1(x)))    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.addbias2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SS_Group_Lasso_Node_Wide_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, lamb_mu, lamb_rho, stride=1, 
                temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()
        self.bn1 = SS_Group_Lasso_VB_BatchNorm2d(inplanes, lamb_mu, lamb_rho, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv1 = SS_Group_Lasso_Node_conv3x3(inplanes, planes, lamb_mu, lamb_rho, stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn2 = SS_Group_Lasso_VB_BatchNorm2d(planes, lamb_mu, lamb_rho, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.conv2 = SS_Group_Lasso_Node_conv3x3(planes, planes, lamb_mu, lamb_rho, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)        

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                SS_Group_Lasso_Node_Conv2d_layer(inplanes, planes, lamb_mu, lamb_rho, 
                                kernel_size=1, stride=stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0),
            )

    def forward(self, x):
        out = self.conv1(F.silu(self.bn1(x)))
        out = self.conv2(F.silu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class SS_Group_Lasso_Node_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=1000, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, lamb_c = 4, lamb_d = 2):
        super().__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.sigma_0 = sigma_0
        self.lamb_c = lamb_c
        self.lamb_d = lamb_d
        self.temp = temp
        self.gamma_prior = gamma_prior
        self.lamb_mu = nn.Parameter(torch.Tensor(1))        
        self.lamb_rho = nn.Parameter(torch.Tensor(1))      
        init.constant_(self.lamb_mu, 1)
        init.constant_(self.lamb_rho, -6.)

        self.kl_lambda = 0.        

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.in_planes = 16
        self.conv1 = SS_Group_Lasso_Node_conv3x3(3,nStages[0], lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, 
                                        temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.layer1 = self._wide_layer(SS_Group_Lasso_Node_Wide_BasicBlock, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(SS_Group_Lasso_Node_Wide_BasicBlock, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(SS_Group_Lasso_Node_Wide_BasicBlock, nStages[3], n, stride=2)
        self.bn1 = SS_Group_Lasso_VB_BatchNorm2d(nStages[3], lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, momentum=0.9, 
                                            temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = Group_Lasso_layer(nStages[3], num_classes, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=sigma_0)
        self.prune_flag = 0
        self.mask = None

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, self.lamb_mu, self.lamb_rho,
                                stride, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        lamb_sigma = torch.log1p(torch.exp(self.lamb_rho))

        self.kl_lambda = -self.lamb_c*torch.log(self.lamb_d) + torch.lgamma(self.lamb_c) - \
                          self.lamb_c*self.lamb_mu + self.lamb_d*torch.exp(self.lamb_mu+(lamb_sigma**2/2)) - \
                          torch.log(lamb_sigma) - 1.41894
        
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
#### Group Lasso without spike-and-slab: Resnet and Wide-Resnet
def Group_Lasso_conv3x3(in_planes, out_planes, lamb_mu, lamb_rho, stride=1, sigma_0 = 1):
    "3x3 convolution with padding"
    return Group_Lasso_Conv2d_layer(in_planes, out_planes, lamb_mu, lamb_rho, kernel_size=3, stride=stride,
                     padding=1, bias=False, sigma_0=sigma_0)

class Group_Lasso_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, block_idx, max_block, lamb_mu, lamb_rho, stride=1, sigma_0 = 1):
        super().__init__()
        self.conv1 = Group_Lasso_conv3x3(inplanes, planes, lamb_mu, lamb_rho, stride = stride, sigma_0 = sigma_0)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = Group_Lasso_conv3x3(planes, planes, lamb_mu, lamb_rho, sigma_0=sigma_0)
        self.stride = stride

        self.addbias1 = Bias2D(inplanes)
        self.addbias2 = Bias2D(planes)

        self._scale = nn.Parameter(torch.ones(1))

        multiplier = (block_idx + 1) ** -(1/6) * max_block **(1/6)

        for m in self.modules():
            if isinstance(m, Group_Lasso_Conv2d_layer):
                _, C, H, W = m.w_mu.shape
                stddev = (C * H * W / 2) ** -.5
                nn.init.normal_(m.w_mu, std = stddev * multiplier)

        self.residual = max_block ** -.5
        self.identity = block_idx ** .5 / (block_idx + 1) ** .5
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            # self.shortcut = LambdaLayer(
            #     lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            self.shortcut = nn.Sequential(
                Bias2D(num_features=inplanes),
                Group_Lasso_Conv2d_layer(inplanes, planes, lamb_mu, lamb_rho, kernel_size=1, stride=stride, sigma_0=sigma_0))
            nn.init.kaiming_normal_(self.shortcut[1].w_mu, a=1)

    def forward(self, x):
        out = self.silu(self.conv1(self.addbias1(x)))
        out = self.conv2(self.addbias2(out))

        out = out.mul(self._scale.mul(self.residual))
        out = torch.add(input=out, alpha=self.identity, other=self.shortcut(x))
        
        out = self.silu(out)

        return out

class Group_Lasso_RescaleNet(nn.Module):

    def __init__(self, depth, num_classes=1000, sigma_0 = 1, lamb_c = 4, lamb_d = 2):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.block_idx = 3*n - 1
        self.max_depth = 3*n
        self.sigma_0 = sigma_0
        self.lamb_c = lamb_c
        self.lamb_d = lamb_d
        self.lamb_mu = nn.Parameter(torch.Tensor(1))        
        self.lamb_rho = nn.Parameter(torch.Tensor(1))      
        init.constant_(self.lamb_mu, 1)
        init.constant_(self.lamb_rho, -6.)

        self.kl_lambda = 0.

        block = Group_Lasso_BasicBlock

        self.inplanes = 16
        self.conv1 = Group_Lasso_Conv2d_layer(3, 16, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho,
                                    kernel_size=3, padding=1, bias=False, sigma_0=sigma_0)
        self.addbias1 = Bias2D(self.inplanes)
        self.silu = nn.SiLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, stride=1)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.addbias2 = Bias2D(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = Group_Lasso_layer(64 * block.expansion, num_classes, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=sigma_0)
        
        nn.init.kaiming_normal_(self.conv1.w_mu)
        nn.init.kaiming_normal_(self.fc.w_mu, a=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, self.block_idx, self.max_depth, self.lamb_mu, self.lamb_rho, stride, sigma_0=self.sigma_0))
        
        self.inplanes = planes * block.expansion
        self.block_idx += 1
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.block_idx, self.max_depth, self.lamb_mu, self.lamb_rho, sigma_0=self.sigma_0))
            self.block_idx += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        lamb_sigma = torch.log1p(torch.exp(self.lamb_rho))

        self.kl_lambda = -self.lamb_c*torch.log(self.lamb_d) + torch.lgamma(self.lamb_c) - \
                          self.lamb_c*self.lamb_mu + self.lamb_d*torch.exp(self.lamb_mu+(lamb_sigma**2/2)) - \
                          torch.log(lamb_sigma) - 1.41894

        x = self.silu(self.addbias1(self.conv1(x)))    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.addbias2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Group_Lasso_Wide_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, lamb_mu, lamb_rho, stride=1, sigma_0 = 1):
        super().__init__()
        self.bn1 = Group_Lasso_VB_BatchNorm2d(inplanes, lamb_mu, lamb_rho, sigma_0=sigma_0)
        self.conv1 = Group_Lasso_conv3x3(inplanes, planes, lamb_mu, lamb_rho, stride, sigma_0=sigma_0)
        self.bn2 = Group_Lasso_VB_BatchNorm2d(planes, lamb_mu, lamb_rho, sigma_0=sigma_0)
        self.conv2 = Group_Lasso_conv3x3(planes, planes, lamb_mu, lamb_rho, sigma_0=sigma_0)        

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                Group_Lasso_Conv2d_layer(inplanes, planes, lamb_mu, lamb_rho, kernel_size=1, stride=stride, sigma_0=sigma_0),
            )

    def forward(self, x):
        out = self.conv1(F.silu(self.bn1(x)))
        out = self.conv2(F.silu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Group_Lasso_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=1000, sigma_0 = 1, lamb_c = 4, lamb_d = 2):
        super().__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.sigma_0 = sigma_0
        self.lamb_c = lamb_c
        self.lamb_d = lamb_d
        self.lamb_mu = nn.Parameter(torch.Tensor(1))        
        self.lamb_rho = nn.Parameter(torch.Tensor(1))      
        init.constant_(self.lamb_mu, 1)
        init.constant_(self.lamb_rho, -6.)

        self.kl_lambda = 0.  

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.in_planes = 16
        self.conv1 = Group_Lasso_conv3x3(3,nStages[0], lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=sigma_0)
        self.layer1 = self._wide_layer(Group_Lasso_Wide_BasicBlock, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(Group_Lasso_Wide_BasicBlock, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(Group_Lasso_Wide_BasicBlock, nStages[3], n, stride=2)
        self.bn1 = Group_Lasso_VB_BatchNorm2d(nStages[3], lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, momentum=0.9, sigma_0=sigma_0)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = Group_Lasso_layer(nStages[3], num_classes, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=sigma_0)
        self.prune_flag = 0
        self.mask = None

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, self.lamb_mu, self.lamb_rho, stride, sigma_0=self.sigma_0))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        lamb_sigma = torch.log1p(torch.exp(self.lamb_rho))

        self.kl_lambda = -self.lamb_c*torch.log(self.lamb_d) + torch.lgamma(self.lamb_c) - \
                          self.lamb_c*self.lamb_mu + self.lamb_d*torch.exp(self.lamb_mu+(lamb_sigma**2/2)) - \
                          torch.log(lamb_sigma) - 1.41894

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.silu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out