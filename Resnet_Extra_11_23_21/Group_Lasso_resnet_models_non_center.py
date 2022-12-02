import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from Group_Lasso_linear_layers_non_center import Group_Lasso_layer
from Group_Lasso_Conv_layers_non_center import SS_Group_Lasso_Node_Conv2d_layer, Group_Lasso_Conv2d_layer
from Group_Lasso_BN_layers_non_center import Group_Lasso_VB_BatchNorm2d

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

    def __init__(self, inplanes, planes, lamb_mu, lamb_rho, stride=1, downsample=None, 
                temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__()
        self.conv1 = SS_Group_Lasso_Node_conv3x3(inplanes, planes, lamb_mu, lamb_rho, stride, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn1 = Group_Lasso_VB_BatchNorm2d(planes, lamb_mu, lamb_rho, sigma_0=sigma_0)
        self.conv2 = SS_Group_Lasso_Node_conv3x3(planes, planes, lamb_mu, lamb_rho, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn2 = Group_Lasso_VB_BatchNorm2d(planes, lamb_mu, lamb_rho, sigma_0=sigma_0)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.silu(out)

        return out

# class SS_Group_Lasso_Node_Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, lamb_mu, lamb_rho, stride=1, downsample=None, 
#                 temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
#         super().__init__()
#         self.conv1 = SS_Group_Lasso_Node_Conv2d_layer(inplanes, planes, lamb_mu, lamb_rho, 
#                                         kernel_size=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
#         self.bn1 = SS_Group_Lasso_VB_BatchNorm2d(planes, lamb_mu, lamb_rho, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
#         self.conv2 = SS_Group_Lasso_Node_Conv2d_layer(planes, planes, lamb_mu, lamb_rho, 
#                             kernel_size=3, stride=stride, padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
#         self.bn2 = SS_Group_Lasso_VB_BatchNorm2d(planes, lamb_mu, lamb_rho, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
#         self.conv3 = SS_Group_Lasso_Node_Conv2d_layer(planes, planes * 4, lamb_mu, lamb_rho, 
#                             kernel_size=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
#         self.bn3 = SS_Group_Lasso_VB_BatchNorm2d(planes * 4, lamb_mu, lamb_rho, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x
#         out = F.silu(self.bn1(self.conv1(x)))
#         out = F.silu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = F.silu(out)

#         return out

class SS_Group_Lasso_Node_ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, lamb_c = 4, lamb_d = 2):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
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

        block = SS_Group_Lasso_Node_Bottleneck if depth >=44 else SS_Group_Lasso_Node_BasicBlock

        self.inplanes = 16
        self.conv1 = SS_Group_Lasso_Node_Conv2d_layer(3, 16, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, 
                                    kernel_size=3, padding=1, bias=False, temp=temp, gamma_prior=gamma_prior, sigma_0=sigma_0)
        self.bn1 = Group_Lasso_VB_BatchNorm2d(16, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=sigma_0)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = Group_Lasso_layer(64 * block.expansion, num_classes, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=sigma_0)

        # for m in self.modules():
        #     if isinstance(m, SS_Group_Lasso_Node_Conv2d_layer):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.w_mu.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, SS_Group_Lasso_VB_BatchNorm2d):
        #         m.weight_mu.data.fill_(1)
        #         m.bias_mu.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SS_Group_Lasso_Node_Conv2d_layer(self.inplanes, planes * block.expansion, self.lamb_mu, self.lamb_rho, 
                                kernel_size=1, stride=stride, bias=False, 
                                temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0),
                Group_Lasso_VB_BatchNorm2d(planes * block.expansion, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=self.sigma_0),
            )

        layers = []
        layers.append(block(self.inplanes, planes,  self.lamb_mu, self.lamb_rho, 
                            stride, downsample, temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,  self.lamb_mu, self.lamb_rho,
                            temp=self.temp, gamma_prior=self.gamma_prior, sigma_0=self.sigma_0))

        return nn.Sequential(*layers)

    def forward(self, x):
        lamb_sigma = torch.log1p(torch.exp(self.lamb_rho))

        self.kl_lambda = -self.lamb_c*torch.log(self.lamb_d) + torch.lgamma(self.lamb_c) - \
                          self.lamb_c*self.lamb_mu + self.lamb_d*torch.exp(self.lamb_mu+(lamb_sigma**2/2)) - \
                          torch.log(lamb_sigma) - 1.41894
        
        x = F.silu(self.bn1(self.conv1(x)))    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
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

    def __init__(self, inplanes, planes, lamb_mu, lamb_rho, stride=1, downsample=None, sigma_0 = 1):
        super().__init__()
        self.conv1 = Group_Lasso_conv3x3(inplanes, planes, lamb_mu, lamb_rho, stride = stride, sigma_0 = sigma_0)
        self.bn1 = Group_Lasso_VB_BatchNorm2d(planes, lamb_mu, lamb_rho, sigma_0=sigma_0)
        self.conv2 = Group_Lasso_conv3x3(planes, planes, lamb_mu, lamb_rho, sigma_0=sigma_0)
        self.bn2 = Group_Lasso_VB_BatchNorm2d(planes, lamb_mu, lamb_rho, sigma_0=sigma_0)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.silu(out)

        return out

class Group_Lasso_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, lamb_mu, lamb_rho, stride=1, downsample=None, sigma_0 = 1):
        super().__init__()
        self.conv1 = Group_Lasso_Conv2d_layer(inplanes, planes, lamb_mu, lamb_rho, kernel_size=1, bias=False, sigma_0=sigma_0)
        self.bn1 = Group_Lasso_VB_BatchNorm2d(planes, lamb_mu, lamb_rho, sigma_0=sigma_0)
        self.conv2 = Group_Lasso_Conv2d_layer(planes, planes, lamb_mu, lamb_rho, kernel_size=3, stride=stride, padding=1, bias=False, sigma_0=sigma_0)
        self.bn2 = Group_Lasso_VB_BatchNorm2d(planes, lamb_mu, lamb_rho, sigma_0=sigma_0)
        self.conv3 = Group_Lasso_Conv2d_layer(planes, planes * 4, lamb_mu, lamb_rho, kernel_size=1, bias=False, sigma_0=sigma_0)
        self.bn3 = Group_Lasso_VB_BatchNorm2d(planes * 4, lamb_mu, lamb_rho, sigma_0=sigma_0)
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

class Group_Lasso_ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, sigma_0 = 1, lamb_c = 4, lamb_d = 2):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
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

        block = Group_Lasso_Bottleneck if depth >=44 else Group_Lasso_BasicBlock

        self.inplanes = 16
        self.conv1 = Group_Lasso_Conv2d_layer(3, 16, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho,
                                    kernel_size=3, padding=1, bias=False, sigma_0=sigma_0)
        self.bn1 = Group_Lasso_VB_BatchNorm2d(16, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=sigma_0)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = Group_Lasso_layer(64 * block.expansion, num_classes, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=sigma_0)
        self.prune_flag = 0
        self.mask = None

        # for m in self.modules():
        #     if isinstance(m, Group_Lasso_Conv2d_layer):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.w_mu.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, Group_Lasso_VB_BatchNorm2d):
        #         m.weight_mu.data.fill_(1)
        #         m.bias_mu.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Group_Lasso_Conv2d_layer(self.inplanes, planes * block.expansion, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, 
                                    kernel_size=1, stride=stride, bias=False, sigma_0=self.sigma_0),
                Group_Lasso_VB_BatchNorm2d(planes * block.expansion, lamb_mu = self.lamb_mu, lamb_rho = self.lamb_rho, sigma_0=self.sigma_0),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.lamb_mu, self.lamb_rho, stride, downsample, sigma_0=self.sigma_0))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.lamb_mu, self.lamb_rho, sigma_0=self.sigma_0))

        return nn.Sequential(*layers)

    def forward(self, x):
        lamb_sigma = torch.log1p(torch.exp(self.lamb_rho))

        self.kl_lambda = -self.lamb_c*torch.log(self.lamb_d) + torch.lgamma(self.lamb_c) - \
                          self.lamb_c*self.lamb_mu + self.lamb_d*torch.exp(self.lamb_mu+(lamb_sigma**2/2)) - \
                          torch.log(lamb_sigma) - 1.41894

        x = F.silu(self.bn1(self.conv1(x)))    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
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