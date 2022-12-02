import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from GHS_linear_layers_non_center_reg import GHS_layer
from GHS_Conv_layers_non_center_reg_new import SS_GHS_Node_Conv2d_layer, GHS_Conv2d_layer
from GHS_BN_layers_non_center_reg import GHS_VB_BatchNorm2d

import numpy as np

__all__ = ['resnet']

##########################################################################################################################################
#### Spike-and-slab node selection with Group Horseshoe: Resnet and Wide-Resnet
def SS_GHS_Node_conv3x3(in_planes, out_planes, fine_tune, stride=1,
                        temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, tau_1 = 1):
    "3x3 convolution with padding"
    
    return SS_GHS_Node_Conv2d_layer(in_planes, out_planes, fine_tune, kernel_size=3, stride=stride,
                     padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)

class SS_GHS_Node_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, fine_tune, stride=1, downsample=None,
                temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, tau_1 = 1):
        super().__init__()
        self.conv1 = SS_GHS_Node_conv3x3(inplanes, planes, fine_tune, stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
        self.bn1 = GHS_VB_BatchNorm2d(planes, sigma_0=sigma_0, tau_1 = tau_1)
        self.conv2 = SS_GHS_Node_conv3x3(planes, planes, fine_tune, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
        self.bn2 = GHS_VB_BatchNorm2d(planes, sigma_0=sigma_0, tau_1 = tau_1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x_dict):
        residual = x_dict
        out = self.bn1(self.conv1(x_dict))
        out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x_dict)
        out = {0:out[0]+residual[0], 1:out[1], 2:out[2]}
        out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}

        return out

# class SS_GHS_Node_Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, 
#                 temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, tau_1 = 1):
#         super().__init__()
#         self.conv1 = SS_GHS_Node_Conv2d_layer(inplanes, planes, kernel_size=1, bias=False, 
#                                                 temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.bn1 = SS_GHS_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.conv2 = SS_GHS_Node_Conv2d_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, 
#                                                 temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.bn2 = SS_GHS_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.conv3 = SS_GHS_Node_Conv2d_layer(planes, planes * 4, kernel_size=1, bias=False, 
#                                                 temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.bn3 = SS_GHS_VB_BatchNorm2d(planes * 4, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x_dict):
#         residual = x_dict
#         out = self.bn1(self.conv1(x_dict))
#         out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}
#         out = self.bn2(self.conv2(out))
#         out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}
#         out = self.bn3(self.conv3(out))
#         if self.downsample is not None:
#             residual = self.downsample(x_dict)
#         out = {0:out[0]+residual[0], 1:out[1], 2:out[2]}
#         out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}

#         return out

class SS_GHS_Node_ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, tau_0 = 1e-5, tau_1 = 1, c_a = 2, c_b = 6, c_reg = 1):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.sigma_0 = sigma_0
        self.c_a = c_a
        self.c_b = c_b
        self.c_reg = c_reg
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.temp = temp
        self.gamma_prior = gamma_prior
        self.sig_a_mu = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_a_mu, 1)
        self.sig_a_rho = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_a_rho, -6.)
        self.sig_b_mu = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_b_mu, 1)
        self.sig_b_rho = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_b_rho, -6.)

        # self.c_mu = nn.Parameter(torch.Tensor(1))
        # init.constant_(self.c_mu, 1)
        # self.c_rho = nn.Parameter(torch.Tensor(1))
        # init.constant_(self.c_rho, -6.)
        # self.kl_c_const = -c_a*torch.log(c_b) + torch.lgamma(c_a) - 1.41894

        self.fine_tune = nn.Parameter(torch.Tensor(1))
        init.constant_(self.fine_tune, 0)

        self.kl_sig_and_c = 0.

        block = SS_GHS_Node_Bottleneck if depth >=44 else SS_GHS_Node_BasicBlock

        self.inplanes = 16
        self.conv1 = SS_GHS_Node_Conv2d_layer(3, 16, fine_tune = self.fine_tune, kernel_size=3, padding=1, 
                                                bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
        self.bn1 = GHS_VB_BatchNorm2d(16, sigma_0=sigma_0, tau_1 = tau_1)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = GHS_layer(64 * block.expansion, num_classes, sigma_0=sigma_0, tau_1 = tau_1)

        # for m in self.modules():
        #     if isinstance(m, SS_GHS_Node_Conv2d_layer):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.w_mu.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, SS_GHS_VB_BatchNorm2d):
        #         m.weight_mu.data.fill_(1)
        #         m.bias_mu.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SS_GHS_Node_Conv2d_layer(self.inplanes, planes * block.expansion, self.fine_tune, kernel_size=1, stride=stride, bias=False, 
                                temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0, tau_1 = self.tau_1),
                GHS_VB_BatchNorm2d(planes * block.expansion, sigma_0=self.sigma_0, tau_1 = self.tau_1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.fine_tune,  
                            stride, downsample, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0, tau_1 = self.tau_1))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.fine_tune, 
                            temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0, tau_1 = self.tau_1))

        return nn.Sequential(*layers)

    def forward(self, x):
        sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
        sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho))

        sigma_sig = 0.5*torch.sqrt(sig_a_sigma**2 + sig_b_sigma**2)
        epsilon_sig = torch.zeros_like(self.sig_a_mu).normal_()
        Global_scale = torch.exp(0.5*(self.sig_a_mu+self.sig_b_mu) + sigma_sig * epsilon_sig)

        # c_sigma = torch.log1p(torch.exp(self.c_rho))
        # epsilon_c = torch.zeros_like(self.c_mu).normal_()
        # c_reg = torch.sqrt(torch.exp(self.c_mu + c_sigma * epsilon_c))

        kl_a_sig = -torch.log(self.tau_0) + torch.exp(self.sig_a_mu + 0.5*sig_a_sigma**2)/self.tau_0 - \
                        0.5*(self.sig_a_mu + 2*torch.log(sig_a_sigma) + 1.69315)
        
        kl_b_sig = torch.exp(0.5*sig_b_sigma**2 - self.sig_b_mu) - \
                        0.5*(2*torch.log(sig_b_sigma) - self.sig_b_mu + 1.69315)

        # kl_c = self.kl_c_const + self.c_a* self.c_mu - torch.log(c_sigma) + \
        #                 self.c_b * torch.exp(0.5*c_sigma**2 - self.c_mu)

        self.kl_sig_and_c = torch.sum(kl_a_sig) + torch.sum(kl_b_sig) #+ torch.sum(kl_c)
        
        x_dict = {0:x, 1:Global_scale, 2:self.c_reg}
        x_dict = self.bn1(self.conv1(x_dict))    # 32x32
        x_dict = {0:F.silu(x_dict[0]), 1:x_dict[1], 2:x_dict[2]}
        x_dict = self.layer1(x_dict)  # 32x32
        x_dict = self.layer2(x_dict)  # 16x16
        x_dict = self.layer3(x_dict)  # 8x8
        x = self.avgpool(x_dict[0])
        x_dict = {0:x.view(x.size(0), -1), 1:x_dict[1], 2:x_dict[2]}
        x_dict = self.fc(x_dict)

        return x_dict[0]

    def fine_tune_flag(self):
        init.constant_(self.fine_tune, 1)

# class SS_GHS_Node_Wide_BasicBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, 
#                 temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, tau_1 = 1):
#         super().__init__()
#         self.bn1 = SS_GHS_VB_BatchNorm2d(inplanes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.conv1 = SS_GHS_Node_conv3x3(inplanes, planes, stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.bn2 = SS_GHS_VB_BatchNorm2d(planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.conv2 = SS_GHS_Node_conv3x3(planes, planes, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)        

#         self.shortcut = nn.Sequential()
#         if stride != 1 or inplanes != planes:
#             self.shortcut = nn.Sequential(
#                 SS_GHS_Node_Conv2d_layer(inplanes, planes, 
#                                 kernel_size=1, stride=stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1),
#             )

#     def forward(self, x_dict):
#         out = self.bn1(x_dict)
#         out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}
#         out = self.bn2(self.conv1(out))
#         out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}
#         out = self.conv2(out)
#         out = {0:out[0]+self.shortcut(x_dict)[0], 1:out[1], 2:out[2]}

#         return out

# class SS_GHS_Node_Wide_ResNet(nn.Module):
#     def __init__(self, depth, widen_factor, num_classes=1000, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, tau_0 = 1e-5, tau_1 = 1, c_a = 2, c_b = 6, c_reg = 1):
#         super().__init__()
#         assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
#         n = (depth - 2) // 6

#         self.sigma_0 = sigma_0
#         self.c_a = c_a
#         self.c_b = c_b
#         self.c_reg = c_reg
#         self.tau_0 = tau_0
#         self.tau_1 = tau_1
#         self.temp = temp
#         self.gamma_prior = gamma_prior
#         self.sig_a_mu = nn.Parameter(torch.Tensor(1))
#         init.constant_(self.sig_a_mu, 1)
#         self.sig_a_rho = nn.Parameter(torch.Tensor(1))
#         init.constant_(self.sig_a_rho, -6.)
#         self.sig_b_mu = nn.Parameter(torch.Tensor(1))
#         init.constant_(self.sig_b_mu, 1)
#         self.sig_b_rho = nn.Parameter(torch.Tensor(1))
#         init.constant_(self.sig_b_rho, -6.)

#         # self.c_mu = nn.Parameter(torch.Tensor(1))
#         # init.constant_(self.c_mu, 1)
#         # self.c_rho = nn.Parameter(torch.Tensor(1))
#         # init.constant_(self.c_rho, -6.)
#         # self.kl_c_const = -c_a*torch.log(c_b) + torch.lgamma(c_a) - 1.41894

#         self.kl_sig_and_c = 0.       

#         assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
#         n = (depth-4)/6
#         k = widen_factor

#         nStages = [16, 16*k, 32*k, 64*k]

#         self.in_planes = 16
#         self.conv1 = SS_GHS_Node_conv3x3(3,nStages[0], temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.layer1 = self._wide_layer(SS_GHS_Node_Wide_BasicBlock, nStages[1], n, stride=1)
#         self.layer2 = self._wide_layer(SS_GHS_Node_Wide_BasicBlock, nStages[2], n, stride=2)
#         self.layer3 = self._wide_layer(SS_GHS_Node_Wide_BasicBlock, nStages[3], n, stride=2)
#         self.bn1 = SS_GHS_VB_BatchNorm2d(nStages[3], momentum=0.9, 
#                                             temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0, tau_1 = tau_1)
#         self.avgpool = nn.AvgPool2d(8)
#         self.linear = GHS_layer(nStages[3], num_classes, sigma_0=sigma_0, tau_1 = tau_1)

#     def _wide_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(int(num_blocks)-1)
#         layers = []

#         for stride in strides:
#             layers.append(block(self.in_planes, planes,
#                                 stride, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0, tau_1 = self.tau_1))
#             self.in_planes = planes

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
#         sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho))

#         sigma_sig = 0.5*torch.sqrt(sig_a_sigma**2 + sig_b_sigma**2)
#         epsilon_sig = torch.zeros_like(self.sig_a_mu).normal_()
#         Global_scale = torch.exp(0.5*(self.sig_a_mu+self.sig_b_mu) + sigma_sig * epsilon_sig)

#         # c_sigma = torch.log1p(torch.exp(self.c_rho))
#         # epsilon_c = torch.zeros_like(self.c_mu).normal_()
#         # c_reg = torch.sqrt(torch.exp(self.c_mu + c_sigma * epsilon_c))

#         kl_a_sig = -torch.log(self.tau_0) + torch.exp(self.sig_a_mu + 0.5*sig_a_sigma**2)/self.tau_0 - \
#                         0.5*(self.sig_a_mu + 2*torch.log(sig_a_sigma) + 1.69315)
        
#         kl_b_sig = torch.exp(0.5*sig_b_sigma**2 - self.sig_b_mu) - \
#                         0.5*(2*torch.log(sig_b_sigma) - self.sig_b_mu + 1.69315)

#         # kl_c = self.kl_c_const + self.c_a* self.c_mu - torch.log(c_sigma) + \
#         #                 self.c_b * torch.exp(0.5*c_sigma**2 - self.c_mu)

#         self.kl_sig_and_c = torch.sum(kl_a_sig) + torch.sum(kl_b_sig) #+ torch.sum(kl_c)
        
#         x_dict = {0:x, 1:Global_scale, 2:self.c_reg}
#         x_dict = self.conv1(x_dict)
#         x_dict = self.layer1(x_dict)
#         x_dict = self.layer2(x_dict)
#         x_dict = self.layer3(x_dict)
#         x_dict = self.bn1(x_dict)
#         x_dict = {0:F.silu(x_dict[0]), 1:x_dict[1], 2:x_dict[2]}
#         x = self.avgpool(x_dict[0])
#         x_dict = {0:x.view(x.size(0), -1), 1:x_dict[1], 2:x_dict[2]}
#         x_dict = self.linear(x_dict)

#         return x_dict[0]

##########################################################################################################################################
#### Group Horseshoe without spike-and-slab: Resnet and Wide-Resnet
def GHS_conv3x3(in_planes, out_planes, stride=1, sigma_0 = 1, tau_1 = 1):
    "3x3 convolution with padding"
    return GHS_Conv2d_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, sigma_0=sigma_0, tau_1 = tau_1)

class GHS_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, sigma_0 = 1, tau_1 = 1):
        super().__init__()
        self.conv1 = GHS_conv3x3(inplanes, planes, stride = stride, sigma_0=sigma_0, tau_1 = tau_1)
        self.bn1 = GHS_VB_BatchNorm2d(planes, sigma_0=sigma_0, tau_1 = tau_1)
        self.conv2 = GHS_conv3x3(planes, planes, sigma_0=sigma_0, tau_1 = tau_1)
        self.bn2 = GHS_VB_BatchNorm2d(planes, sigma_0=sigma_0, tau_1 = tau_1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x_dict):
        residual = x_dict
        out = self.bn1(self.conv1(x_dict))
        out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x_dict)
        out = {0:out[0]+residual[0], 1:out[1], 2:out[2]}
        out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}

        return out

class GHS_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, sigma_0 = 1, tau_1 = 1):
        super().__init__()
        self.conv1 = GHS_Conv2d_layer(inplanes, planes,  kernel_size=1, bias=False, sigma_0=sigma_0, tau_1 = tau_1)
        self.bn1 = GHS_VB_BatchNorm2d(planes, sigma_0=sigma_0, tau_1 = tau_1)
        self.conv2 = GHS_Conv2d_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, sigma_0=sigma_0, tau_1 = tau_1)
        self.bn2 = GHS_VB_BatchNorm2d(planes, sigma_0=sigma_0, tau_1 = tau_1)
        self.conv3 = GHS_Conv2d_layer(planes, planes * 4, kernel_size=1, bias=False, sigma_0=sigma_0, tau_1 = tau_1)
        self.bn3 = GHS_VB_BatchNorm2d(planes * 4, sigma_0=sigma_0, tau_1 = tau_1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x_dict):
        residual = x_dict
        out = self.bn1(self.conv1(x_dict))
        out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}
        out = self.bn2(self.conv2(out))
        out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x_dict)
        out = {0:out[0]+residual[0], 1:out[1], 2:out[2]}
        out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}

        return out

class GHS_ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, sigma_0 = 1, tau_0 = 1e-5, tau_1 = 1, c_a = 2, c_b = 6, c_reg = 1):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.sigma_0 = sigma_0
        self.c_a = c_a
        self.c_b = c_b
        self.c_reg = c_reg
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.sig_a_mu = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_a_mu, 1)
        self.sig_a_rho = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_a_rho, -6.)
        self.sig_b_mu = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_b_mu, 1)
        self.sig_b_rho = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_b_rho, -6.)

        # self.c_mu = nn.Parameter(torch.Tensor(1))
        # init.constant_(self.c_mu, 1)
        # self.c_rho = nn.Parameter(torch.Tensor(1))
        # init.constant_(self.c_rho, -6.)
        # self.kl_c_const = -c_a*torch.log(c_b) + torch.lgamma(c_a) - 1.41894

        self.kl_sig_and_c = 0.

        block = GHS_Bottleneck if depth >=44 else GHS_BasicBlock

        self.inplanes = 16
        self.conv1 = GHS_Conv2d_layer(3, 16, kernel_size=3, padding=1, bias=False, sigma_0=sigma_0, tau_1 = tau_1)
        self.bn1 = GHS_VB_BatchNorm2d(16, sigma_0=sigma_0, tau_1 = tau_1)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = GHS_layer(64 * block.expansion, num_classes, sigma_0=sigma_0, tau_1 = tau_1)

        # for m in self.modules():
        #     if isinstance(m, GHS_Conv2d_layer):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.w_mu.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, GHS_VB_BatchNorm2d):
        #         m.weight_mu.data.fill_(1)
        #         m.bias_mu.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                GHS_Conv2d_layer(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, sigma_0=self.sigma_0, tau_1 = self.tau_1),
                GHS_VB_BatchNorm2d(planes * block.expansion, sigma_0=self.sigma_0, tau_1 = self.tau_1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, sigma_0=self.sigma_0, tau_1 = self.tau_1))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, sigma_0=self.sigma_0, tau_1 = self.tau_1))

        return nn.Sequential(*layers)

    def forward(self, x):
        sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
        sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho))

        sigma_sig = 0.5*torch.sqrt(sig_a_sigma**2 + sig_b_sigma**2)
        epsilon_sig = torch.zeros_like(self.sig_a_mu).normal_()
        Global_scale = torch.exp(0.5*(self.sig_a_mu+self.sig_b_mu) + sigma_sig * epsilon_sig)

        # c_sigma = torch.log1p(torch.exp(self.c_rho))
        # epsilon_c = torch.zeros_like(self.c_mu).normal_()
        # c_reg = torch.sqrt(torch.exp(self.c_mu + c_sigma * epsilon_c))

        kl_a_sig = -torch.log(self.tau_0) + torch.exp(self.sig_a_mu + 0.5*sig_a_sigma**2)/self.tau_0 - \
                        0.5*(self.sig_a_mu + 2*torch.log(sig_a_sigma) + 1.69315)
        
        kl_b_sig = torch.exp(0.5*sig_b_sigma**2 - self.sig_b_mu) - \
                        0.5*(2*torch.log(sig_b_sigma) - self.sig_b_mu + 1.69315)

        # kl_c = self.kl_c_const + self.c_a* self.c_mu - torch.log(c_sigma) + \
        #                 self.c_b * torch.exp(0.5*c_sigma**2 - self.c_mu)

        self.kl_sig_and_c = torch.sum(kl_a_sig) + torch.sum(kl_b_sig) #+ torch.sum(kl_c)
        
        x_dict = {0:x, 1:Global_scale, 2:self.c_reg}
        x_dict = self.bn1(self.conv1(x_dict))    # 32x32
        x_dict = {0:F.silu(x_dict[0]), 1:x_dict[1], 2:x_dict[2]}
        x_dict = self.layer1(x_dict)  # 32x32
        x_dict = self.layer2(x_dict)  # 16x16
        x_dict = self.layer3(x_dict)  # 8x8
        x = self.avgpool(x_dict[0])
        x_dict = {0:x.view(x.size(0), -1), 1:x_dict[1], 2:x_dict[2]}
        x_dict = self.fc(x_dict)

        return x_dict[0]

class GHS_Wide_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, sigma_0 = 1, tau_1 = 1):
        super().__init__()
        self.bn1 = GHS_VB_BatchNorm2d(inplanes, sigma_0=sigma_0, tau_1 = tau_1)
        self.conv1 = GHS_conv3x3(inplanes, planes, stride, sigma_0=sigma_0, tau_1 = tau_1)
        self.bn2 = GHS_VB_BatchNorm2d(planes, sigma_0=sigma_0, tau_1 = tau_1)
        self.conv2 = GHS_conv3x3(planes, planes, sigma_0=sigma_0, tau_1 = tau_1)        

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                GHS_Conv2d_layer(inplanes, planes, 
                                    kernel_size=1, stride=stride, sigma_0=sigma_0, tau_1 = tau_1),
            )

    def forward(self, x_dict):
        out = self.bn1(x_dict)
        out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}
        out = self.bn2(self.conv1(out))
        out = {0:F.silu(out[0]), 1:out[1], 2:out[2]}
        out = self.conv2(out)
        out = {0:out[0]+self.shortcut(x_dict)[0], 1:out[1], 2:out[2]}

        return out

class GHS_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=1000, sigma_0 = 1, tau_0 = 1e-5, tau_1 = 1, c_a = 2, c_b = 6, c_reg = 1):
        super().__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.sigma_0 = sigma_0
        self.c_a = c_a
        self.c_b = c_b
        self.c_reg = c_reg
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.sig_a_mu = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_a_mu, 1)
        self.sig_a_rho = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_a_rho, -6.)
        self.sig_b_mu = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_b_mu, 1)
        self.sig_b_rho = nn.Parameter(torch.Tensor(1))
        init.constant_(self.sig_b_rho, -6.)

        # self.c_mu = nn.Parameter(torch.Tensor(1))
        # init.constant_(self.c_mu, 1)
        # self.c_rho = nn.Parameter(torch.Tensor(1))
        # init.constant_(self.c_rho, -6.)
        # self.kl_c_const = -c_a*torch.log(c_b) + torch.lgamma(c_a) - 1.41894

        self.kl_sig_and_c = 0.

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.in_planes = 16
        self.conv1 = GHS_conv3x3(3,nStages[0], sigma_0=sigma_0, tau_1 = tau_1)
        self.layer1 = self._wide_layer(GHS_Wide_BasicBlock, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(GHS_Wide_BasicBlock, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(GHS_Wide_BasicBlock, nStages[3], n, stride=2)
        self.bn1 = GHS_VB_BatchNorm2d(nStages[3], momentum=0.9, sigma_0=sigma_0, tau_1 = tau_1)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = GHS_layer(nStages[3], num_classes, sigma_0=sigma_0, tau_1 = tau_1)

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sigma_0=self.sigma_0, tau_1 = self.tau_1))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
        sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho))

        sigma_sig = 0.5*torch.sqrt(sig_a_sigma**2 + sig_b_sigma**2)
        epsilon_sig = torch.zeros_like(self.sig_a_mu).normal_()
        Global_scale = torch.exp(0.5*(self.sig_a_mu+self.sig_b_mu) + sigma_sig * epsilon_sig)

        # c_sigma = torch.log1p(torch.exp(self.c_rho))
        # epsilon_c = torch.zeros_like(self.c_mu).normal_()
        # c_reg = torch.sqrt(torch.exp(self.c_mu + c_sigma * epsilon_c))

        kl_a_sig = -torch.log(self.tau_0) + torch.exp(self.sig_a_mu + 0.5*sig_a_sigma**2)/self.tau_0 - \
                        0.5*(self.sig_a_mu + 2*torch.log(sig_a_sigma) + 1.69315)
        
        kl_b_sig = torch.exp(0.5*sig_b_sigma**2 - self.sig_b_mu) - \
                        0.5*(2*torch.log(sig_b_sigma) - self.sig_b_mu + 1.69315)

        # kl_c = self.kl_c_const + self.c_a* self.c_mu - torch.log(c_sigma) + \
        #                 self.c_b * torch.exp(0.5*c_sigma**2 - self.c_mu)

        self.kl_sig_and_c = torch.sum(kl_a_sig) + torch.sum(kl_b_sig) #+ torch.sum(kl_c)
        
        x_dict = {0:x, 1:Global_scale, 2:self.c_reg}
        x_dict = self.conv1(x_dict)
        x_dict = self.layer1(x_dict)
        x_dict = self.layer2(x_dict)
        x_dict = self.layer3(x_dict)
        x_dict = self.bn1(x_dict)
        x_dict = {0:F.silu(x_dict[0]), 1:x_dict[1], 2:x_dict[2]}
        x = self.avgpool(x_dict[0])
        x_dict = {0:x.view(x.size(0), -1), 1:x_dict[1], 2:x_dict[2]}
        x_dict = self.linear(x_dict)

        return x_dict[0]