import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.utils import _single, _pair, _triple
torch.set_default_dtype(torch.float32)

def sigmoid(z):
    return 1. / (1 + torch.exp(-z))

def logit(z):
    return torch.log(z/(1.-z))

def gumbel_softmax(logits, U, temp, hard=False, eps=1e-10):
    z = logits + torch.log(U + eps) - torch.log(1 - U + eps)
    y = 1 / (1 + torch.exp(- z / temp))
    if not hard:
        return y
    y_hard = (y > 0.5).float()
    y_hard = (y_hard - y).detach() + y
    return y_hard

##########################################################################################################################################
#### Spike-and-slab node selection with group horseshoe Layer
class SS_GHS_Node_VB_ConvNd(torch.nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                    padding, dilation, transposed, output_padding, 
                    groups, bias, padding_mode, clip_std):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.clip_std = clip_std

        if transposed:
            self.w_mu = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.w_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_rho = nn.Parameter(torch.Tensor(out_channels))            
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
        self.tau_a_mu = nn.Parameter(torch.Tensor(out_channels))
        self.tau_b_mu = nn.Parameter(torch.Tensor(out_channels))
        self.tau_a_rho = nn.Parameter(torch.Tensor(out_channels))
        self.tau_b_rho = nn.Parameter(torch.Tensor(out_channels))
        self.theta = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

        # initialize kl for the hidden layer
        self.kl = 0

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        # init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.constant_(self.w_rho, -6.)        
        if self.v_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.v_mu, -bound, bound)
            init.constant_(self.v_rho, -6.)
        init.uniform_(self.tau_a_mu, -0.6, 0.6)
        init.uniform_(self.tau_b_mu, -0.6, 0.6)
        init.constant_(self.tau_a_rho, -6.)
        init.constant_(self.tau_b_rho, -6.)
        init.constant_(self.theta, logit(torch.tensor(0.99)))

    def clip_standard_dev(self):
        if self.clip_std:
            self.w_rho.data.clamp_(max=math.log(math.expm1(self.clip_std)))
            if self.v_mu is not None:
                self.v_rho.data.clamp_(max=math.log(math.expm1(self.clip_std)))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias_mu is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class SS_GHS_Node_Conv2d_layer(SS_GHS_Node_VB_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', clip_std=None, 
                 temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, tau_1 = 1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, clip_std)

        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))
        self.register_buffer('tau_1', torch.as_tensor(tau_1))
        self.register_buffer('temp', torch.as_tensor(temp))
        self.register_buffer('gamma_prior', torch.as_tensor(gamma_prior))

    def conv2d_forward(self, input, w, v):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                w, v, self.stride,
                                _pair(0), self.dilation, self.groups)
        return F.conv2d(input, w, v, self.stride,
                            self.padding, self.dilation, self.groups)

    def forward(self, input_dict):
        """
            For one Monte Carlo (MC) sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        Global_scale = input_dict[1]
        u = torch.zeros_like(self.theta).uniform_(0.0, 1.0)
        z = gumbel_softmax(self.theta, u, self.temp, hard=True)
        gamma = sigmoid(self.theta)
        if self.transposed:
            w_z = z.expand(
                            self.in_channels, *self.kernel_size, self.out_channels // self.groups).permute(0,3,1,2)
            w_gamma = gamma.expand(
                                self.in_channels, *self.kernel_size, self.out_channels // self.groups).permute(0,3,1,2)
        else:
            w_z = z.expand( 
                            self.in_channels // self.groups, *self.kernel_size, self.out_channels).permute(3,0,1,2)
            w_gamma = gamma.expand( 
                                self.in_channels // self.groups, *self.kernel_size, self.out_channels).permute(3,0,1,2)

        epsilon_w = torch.zeros_like(self.w_mu).normal_()            
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        
        sigma_a_tau = torch.log1p(torch.exp(self.tau_a_rho))
        sigma_b_tau = torch.log1p(torch.exp(self.tau_b_rho))
        sigma_tau = 0.5*torch.sqrt(sigma_a_tau**2 + sigma_b_tau**2)
        epsilon_tau = torch.zeros_like(self.tau_a_mu).normal_()
        GHS_scale = Global_scale * \
                        torch.exp(0.5*(self.tau_a_mu+self.tau_b_mu) + sigma_tau * epsilon_tau)
        if self.transposed:
            GHS_scale_w = GHS_scale.expand(
                            self.in_channels, *self.kernel_size, self.out_channels // self.groups).permute(0,3,1,2)
        else:
            GHS_scale_w = GHS_scale.expand( 
                            self.in_channels // self.groups, *self.kernel_size, self.out_channels).permute(3,0,1,2)

        self.w = w_z * GHS_scale_w * (self.w_mu + sigma_w * epsilon_w)
        w_1 = w_gamma * GHS_scale_w * (self.w_mu + sigma_w * epsilon_w)
        
        if self.v_mu is not None:
            v_z = z
            v_gamma = gamma
            epsilon_v = torch.zeros_like(self.v_mu).normal_()
            sigma_v = torch.log1p(torch.exp(self.v_rho))
            self.v = v_z * GHS_scale * (self.v_mu + sigma_v * epsilon_v)
            v_1 = v_gamma * GHS_scale * (self.v_mu + sigma_v * epsilon_v)
        else:
            self.v = None
            v_1 = None

        if self.training:
            # gamma = sigmoid(self.theta)
            
            # kl_gamma = gamma * (torch.log(gamma) - torch.log(self.gamma_prior)) + \
            #     (1 - gamma) * (torch.log(1 - gamma) - torch.log(1 - self.gamma_prior)) 
        
            kl_w = torch.log(self.sigma_0) - torch.log(sigma_w) - 0.5 + \
                    0.5*((sigma_w**2+self.w_mu**2)/self.sigma_0**2)

            kl_a_tau = -torch.log(self.tau_1) + torch.exp(self.tau_a_mu + 0.5*sigma_a_tau**2)/self.tau_1 - \
                        0.5*(self.tau_a_mu + 2*torch.log(sigma_a_tau) + 1.69315)

            kl_b_tau = torch.exp(0.5*sigma_b_tau**2 - self.tau_b_mu) - \
                            0.5*(2*torch.log(sigma_b_tau) - self.tau_b_mu + 1.69315)

            if self.v_mu is not None:
                kl_v = torch.log(self.sigma_0) - torch.log(sigma_v) - 0.5 + \
                        0.5*((sigma_v**2+self.v_mu**2)/self.sigma_0**2)
                
                # self.kl = torch.sum(kl_gamma) + \
                #             torch.sum(kl_w) + torch.sum(kl_v) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)
                self.kl = torch.sum(kl_w) + torch.sum(kl_v) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)
            else:
                # self.kl = torch.sum(kl_gamma) + \
                #             torch.sum(kl_w) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)   
                self.kl = torch.sum(kl_w) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)          
        
        return {0:self.conv2d_forward(input_dict[0], w_1, v_1), 1:Global_scale}

##########################################################################################################################################
#### Group Horseshoe without spike-and-slab Layer
class GHS_VB_ConvNd(torch.nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, clip_std):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.clip_std = clip_std

        if transposed:
            self.w_mu = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.w_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_rho = nn.Parameter(torch.Tensor(out_channels))            
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
        self.tau_a_mu = nn.Parameter(torch.Tensor(out_channels))
        self.tau_b_mu = nn.Parameter(torch.Tensor(out_channels))
        self.tau_a_rho = nn.Parameter(torch.Tensor(out_channels))
        self.tau_b_rho = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

        # initialize kl for the hidden layer
        self.kl = 0

    def reset_parameters(self):
        # n = self.in_channels
        # for k in self.kernel_size:
        #     n *= k
        # stdv = 1. / math.sqrt(n)
        # self.w_mu.data.uniform_(-stdv, stdv)
        init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        # init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.constant_(self.w_rho, -6.)        
        if self.v_mu is not None:
            # self.v_mu.data.fill_(0)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.v_mu, -bound, bound)
            init.constant_(self.v_rho, -6.)
        init.uniform_(self.tau_a_mu, -0.6, 0.6)
        init.uniform_(self.tau_b_mu, -0.6, 0.6)
        init.constant_(self.tau_a_rho, -6.)
        init.constant_(self.tau_b_rho, -6.)

    def clip_standard_dev(self):
        if self.clip_std:
            self.w_rho.data.clamp_(max=math.log(math.expm1(self.clip_std)))
            if self.v_mu is not None:
                self.v_rho.data.clamp_(max=math.log(math.expm1(self.clip_std)))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias_mu is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class GHS_Conv2d_layer(GHS_VB_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', clip_std=None, sigma_0 =1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, clip_std)

        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))

    def conv2d_forward(self, input, w, v):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                w, v, self.stride,
                                _pair(0), self.dilation, self.groups)
        return F.conv2d(input, w, v, self.stride,
                            self.padding, self.dilation, self.groups)

    def forward(self, input_dict):
        """
            For one Monte Carlo (MC) sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        Global_scale = input_dict[1]
        epsilon_w = torch.zeros_like(self.w_mu).normal_()            
        sigma_w = torch.log1p(torch.exp(self.w_rho))

        sigma_a_tau = torch.log1p(torch.exp(self.tau_a_rho))
        sigma_b_tau = torch.log1p(torch.exp(self.tau_b_rho))
        sigma_tau = 0.5*torch.sqrt(sigma_a_tau**2 + sigma_b_tau**2)
        epsilon_tau = torch.zeros_like(self.tau_a_mu).normal_()
        GHS_scale = Global_scale * \
                        torch.exp(0.5*(self.tau_a_mu+self.tau_b_mu) + sigma_tau * epsilon_tau)
        if self.transposed:
            GHS_scale_w = GHS_scale.expand(
                            self.in_channels, *self.kernel_size, self.out_channels // self.groups).permute(0,3,1,2)
        else:
            GHS_scale_w = GHS_scale.expand( 
                            self.in_channels // self.groups, *self.kernel_size, self.out_channels).permute(3,0,1,2)

        self.w = GHS_scale_w * (self.w_mu + sigma_w * epsilon_w)
        
        if self.v_mu is not None:
            epsilon_v = torch.zeros_like(self.v_mu).normal_()
            sigma_v = torch.log1p(torch.exp(self.v_rho))
            self.v = GHS_scale * (self.v_mu + sigma_v * epsilon_v)
        else:
            self.v = None 

        if self.training:        
            kl_w = torch.log(self.sigma_0) - torch.log(sigma_w) - 0.5 + \
                    0.5*((sigma_w**2+self.w_mu**2)/self.sigma_0**2)

            kl_a_tau = torch.exp(self.tau_a_mu + 0.5*sigma_a_tau**2) - \
                        0.5*(self.tau_a_mu + 2*torch.log(sigma_a_tau) + 1.69315)

            kl_b_tau = torch.exp(0.5*sigma_b_tau**2 - self.tau_b_mu) - \
                            0.5*(2*torch.log(sigma_b_tau) - self.tau_b_mu + 1.69315)

            if self.v_mu is not None:
                kl_v = torch.log(self.sigma_0) - torch.log(sigma_v) - 0.5 + \
                        0.5*((sigma_v**2+self.v_mu**2)/self.sigma_0**2)
                
                self.kl = torch.sum(kl_w) + torch.sum(kl_v) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)
            else:
                self.kl = torch.sum(kl_w) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)                     

        return {0:self.conv2d_forward(input_dict[0], self.w, self.v), 1:Global_scale}