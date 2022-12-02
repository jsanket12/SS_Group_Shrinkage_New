import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.utils import _single, _pair, _triple

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
#### Spike-and-slab edge selection with horseshoe Layer
class SSHS_Edge_VB_ConvNd(torch.nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                    padding, dilation, transposed, output_padding, 
                    groups, bias, padding_mode):
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

        if transposed:
            self.w_mu = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_tau_a_mu = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_tau_b_mu = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_tau_a_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_tau_b_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_theta = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.w_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_tau_a_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_tau_b_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_tau_a_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_tau_b_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_theta = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_rho = nn.Parameter(torch.Tensor(out_channels))
            self.v_tau_a_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_tau_b_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_tau_a_rho = nn.Parameter(torch.Tensor(out_channels))
            self.v_tau_b_rho = nn.Parameter(torch.Tensor(out_channels))
            self.v_theta = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
            self.register_parameter('v_tau_a_mu', None)
            self.register_parameter('v_tau_b_mu', None)
            self.register_parameter('v_tau_a_rho', None)
            self.register_parameter('v_tau_b_rho', None)
            self.register_parameter('v_theta', None)
        self.reset_parameters()

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

        # initialize kl for the hidden layer
        self.kl = 0

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        # init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        init.constant_(self.w_rho, -6.)
        init.uniform_(self.w_tau_a_mu, -0.6, 0.6)
        init.uniform_(self.w_tau_b_mu, -0.6, 0.6)
        init.constant_(self.w_tau_a_rho, -6.)
        init.constant_(self.w_tau_b_rho, -6.)
        init.constant_(self.w_theta, logit(torch.tensor(0.99)))
        if self.v_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.v_mu, -bound, bound)
            init.constant_(self.v_rho, -6.)
            init.uniform_(self.v_tau_a_mu, -0.6, 0.6)
            init.uniform_(self.v_tau_b_mu, -0.6, 0.6)
            init.constant_(self.v_tau_a_rho, -6.)
            init.constant_(self.v_tau_b_rho, -6.)
            init.constant_(self.v_theta, logit(torch.tensor(0.99)))

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

class SSHS_Edge_Conv2d_layer(SSHS_Edge_VB_ConvNd):
    def __init__(self, in_channels, out_channels, sig_a_mu, sig_a_rho, sig_b_mu, sig_b_rho, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, tau_1 = 1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.sig_a_mu = sig_a_mu 
        self.sig_a_rho = sig_a_rho
        self.sig_b_mu = sig_b_mu 
        self.sig_b_rho = sig_b_rho
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
        c_reg = input_dict[2]
        u_w = torch.zeros_like(self.w_mu).uniform_(0.0, 1.0)        
        w_z = gumbel_softmax(self.w_theta, u_w, self.temp, hard=True)        
        epsilon_w = torch.zeros_like(self.w_mu).normal_()
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        self.w = w_z * (self.w_mu + sigma_w * epsilon_w)

        if self.v_mu is not None:
            u_v = torch.zeros_like(self.v_mu).uniform_(0.0, 1.0)
            v_z = gumbel_softmax(self.v_theta, u_v, self.temp, hard=True)
            epsilon_v = torch.zeros_like(self.v_mu).normal_()
            sigma_v = torch.log1p(torch.exp(self.v_rho))
            self.v = v_z * self.v_mu + sigma_v * epsilon_v
        else:
            self.v = None

        if self.training:
            w_gamma = sigmoid(self.w_theta)
            sigma_w_a_tau = torch.log1p(torch.exp(self.w_tau_a_rho))
            sigma_w_b_tau = torch.log1p(torch.exp(self.w_tau_b_rho))
            sigma_w_tau = 0.5*torch.sqrt(sigma_w_a_tau**2 + sigma_w_b_tau**2)
            epsilon_w_tau = torch.zeros_like(self.w_tau_a_mu).normal_()
            GHS_scale_w = Global_scale * \
                        torch.sqrt(torch.exp(0.5*(self.w_tau_a_mu+self.w_tau_b_mu) + sigma_w_tau * epsilon_w_tau))
            sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
            sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho))

            kl_w_gamma = w_gamma * (torch.log(w_gamma) - torch.log(self.gamma_prior)) + \
                        (1 - w_gamma) * (torch.log(1 - w_gamma) - torch.log(1 - self.gamma_prior)) 
        
            kl_w = w_gamma * (torch.log(self.sigma_0) + torch.log(c_reg) - torch.log(sigma_w) - 0.5 + \
                    0.5*(self.w_tau_a_mu + self.w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) - 0.5*torch.log(c_reg**2  + GHS_scale_w**2) + \
                    0.5*((sigma_w**2+self.w_mu**2)/self.sigma_0**2)* \
                    (torch.exp(-(self.w_tau_a_mu + self.w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                        0.5*(sigma_w_a_tau**2 + sigma_w_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2)) + 1/(c_reg**2)))


            kl_w_a_tau = -torch.log(self.tau_1) + torch.exp(self.w_tau_a_mu + 0.5*sigma_w_a_tau**2)/self.tau_1 - \
                        0.5*(self.w_tau_a_mu + 2*torch.log(sigma_w_a_tau) + 1.69315)

            kl_w_b_tau = torch.exp(0.5*sigma_w_b_tau**2 - self.w_tau_b_mu) - \
                        0.5*(2*torch.log(sigma_w_b_tau) - self.w_tau_b_mu + 1.69315)

            if self.v_mu is not None:
                v_gamma = sigmoid(self.v_theta)
                sigma_v_a_tau = torch.log1p(torch.exp(self.v_tau_a_rho))
                sigma_v_b_tau = torch.log1p(torch.exp(self.v_tau_b_rho))
                sigma_v_tau = 0.5*torch.sqrt(sigma_v_a_tau**2 + sigma_v_b_tau**2)
                epsilon_v_tau = torch.zeros_like(self.v_tau_a_mu).normal_()
                GHS_scale_v = Global_scale * \
                                torch.sqrt(torch.exp(0.5*(self.v_tau_a_mu+self.v_tau_b_mu) + sigma_v_tau * epsilon_v_tau))
            
                
                kl_v_gamma = v_gamma * (torch.log(v_gamma) - torch.log(self.gamma_prior)) + \
                            (1 - v_gamma) * (torch.log(1 - v_gamma) - torch.log(1 - self.gamma_prior)) 

                kl_v = v_gamma * (torch.log(self.sigma_0) + torch.log(c_reg) - torch.log(sigma_v) - 0.5 + \
                        0.5*(self.v_tau_a_mu + self.v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) - 0.5*torch.log(c_reg**2  + GHS_scale_v**2) + \
                        0.5*((sigma_v**2+self.v_mu**2)/self.sigma_0**2)* \
                        (torch.exp(-(self.v_tau_a_mu + self.v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                            0.5*(sigma_v_a_tau**2 + sigma_v_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2)) + 1/(c_reg**2)))
                
                kl_v_a_tau = -torch.log(self.tau_1) + torch.exp(self.v_tau_a_mu + 0.5*sigma_v_a_tau**2)/self.tau_1 - \
                                0.5*(self.v_tau_a_mu + 2*torch.log(sigma_v_a_tau) + 1.69315)

                kl_v_b_tau = torch.exp(0.5*sigma_v_b_tau**2 - self.v_tau_b_mu) - \
                                0.5*(2*torch.log(sigma_v_b_tau) - self.v_tau_b_mu + 1.69315)
                
                self.kl = torch.sum(kl_w_gamma) + torch.sum(kl_v_gamma) + torch.sum(kl_w) + torch.sum(kl_v) + \
                            torch.sum(kl_w_a_tau) + torch.sum(kl_v_a_tau) + torch.sum(kl_w_b_tau) + torch.sum(kl_v_b_tau)
            else:
                self.kl = torch.sum(kl_w_gamma) + torch.sum(kl_w) + torch.sum(kl_w_a_tau) + torch.sum(kl_w_b_tau)

        return {0:self.conv2d_forward(input_dict[0], self.w, self.v), 1:Global_scale, 2:c_reg}

##########################################################################################################################################
#### Horseshoe without spike-and-slab Layer
class HS_VB_ConvNd(torch.nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
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

        if transposed:
            self.w_mu = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_tau_a_mu = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_tau_b_mu = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_tau_a_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_tau_b_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.w_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_tau_a_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_tau_b_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_tau_a_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_tau_b_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_rho = nn.Parameter(torch.Tensor(out_channels))
            self.v_tau_a_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_tau_b_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_tau_a_rho = nn.Parameter(torch.Tensor(out_channels))
            self.v_tau_b_rho = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
            self.register_parameter('v_tau_a_mu', None)
            self.register_parameter('v_tau_b_mu', None)
            self.register_parameter('v_tau_a_rho', None)
            self.register_parameter('v_tau_b_rho', None)
        self.reset_parameters()

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

        # initialize kl for the hidden layer
        self.kl = 0

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        # init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        init.constant_(self.w_rho, -6.)
        init.uniform_(self.w_tau_a_mu, -0.6, 0.6)
        init.uniform_(self.w_tau_b_mu, -0.6, 0.6)
        init.constant_(self.w_tau_a_rho, -6.)
        init.constant_(self.w_tau_b_rho, -6.)
        if self.v_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.v_mu, -bound, bound)
            init.constant_(self.v_rho, -6.)
            init.uniform_(self.v_tau_a_mu, -0.6, 0.6)
            init.uniform_(self.v_tau_b_mu, -0.6, 0.6)
            init.constant_(self.v_tau_a_rho, -6.)
            init.constant_(self.v_tau_b_rho, -6.)

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

class HS_Conv2d_layer(HS_VB_ConvNd):
    def __init__(self, in_channels, out_channels, sig_a_mu, sig_a_rho, sig_b_mu, sig_b_rho, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', sigma_0 =1, tau_1 = 1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.sig_a_mu = sig_a_mu 
        self.sig_a_rho = sig_a_rho
        self.sig_b_mu = sig_b_mu 
        self.sig_b_rho = sig_b_rho
        self.register_buffer('tau_1', torch.as_tensor(tau_1))
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
        c_reg = input_dict[2]
        epsilon_w = torch.zeros_like(self.w_mu).normal_()
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        self.w = self.w_mu + sigma_w * epsilon_w

        if self.v_mu is not None:
            epsilon_v = torch.zeros_like(self.v_mu).normal_()
            sigma_v = torch.log1p(torch.exp(self.v_rho))            
            self.v = self.v_mu + sigma_v * epsilon_v
        else:
            self.v = None

        if self.training:
            sigma_w_a_tau = torch.log1p(torch.exp(self.w_tau_a_rho))
            sigma_w_b_tau = torch.log1p(torch.exp(self.w_tau_b_rho))
            sigma_w_tau = 0.5*torch.sqrt(sigma_w_a_tau**2 + sigma_w_b_tau**2)
            epsilon_w_tau = torch.zeros_like(self.w_tau_a_mu).normal_()
            GHS_scale_w = Global_scale * \
                            torch.sqrt(torch.exp(0.5*(self.w_tau_a_mu+self.w_tau_b_mu) + sigma_w_tau * epsilon_w_tau))
            sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
            sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho)) 

            kl_w = torch.log(self.sigma_0) + torch.log(c_reg) - torch.log(sigma_w) - 0.5 + \
                    0.5*(self.w_tau_a_mu + self.w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) - 0.5*torch.log(c_reg**2  + GHS_scale_w**2) + \
                    0.5*((sigma_w**2+self.w_mu**2)/self.sigma_0**2)* \
                    (torch.exp(-(self.w_tau_a_mu + self.w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                        0.5*(sigma_w_a_tau**2 + sigma_w_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2)) + 1/(c_reg**2))

            
            kl_w_a_tau = -torch.log(self.tau_1) + torch.exp(self.w_tau_a_mu + 0.5*sigma_w_a_tau**2)/self.tau_1 - \
                        0.5*(self.w_tau_a_mu + 2*torch.log(sigma_w_a_tau) + 1.69315)

            kl_w_b_tau = torch.exp(0.5*sigma_w_b_tau**2 - self.w_tau_b_mu) - \
                        0.5*(2*torch.log(sigma_w_b_tau) - self.w_tau_b_mu + 1.69315)

            if self.v_mu is not None:
                sigma_v_a_tau = torch.log1p(torch.exp(self.v_tau_a_rho))
                sigma_v_b_tau = torch.log1p(torch.exp(self.v_tau_b_rho))
                sigma_v_tau = 0.5*torch.sqrt(sigma_v_a_tau**2 + sigma_v_b_tau**2)
                epsilon_v_tau = torch.zeros_like(self.v_tau_a_mu).normal_()
                GHS_scale_v = Global_scale * \
                                torch.sqrt(torch.exp(0.5*(self.v_tau_a_mu+self.v_tau_b_mu) + sigma_v_tau * epsilon_v_tau))

                kl_v = torch.log(self.sigma_0) + torch.log(c_reg) - torch.log(sigma_v) - 0.5 + \
                        0.5*(self.v_tau_a_mu + self.v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) - 0.5*torch.log(c_reg**2  + GHS_scale_v**2) + \
                        0.5*((sigma_v**2+self.v_mu**2)/self.sigma_0**2)* \
                        (torch.exp(-(self.v_tau_a_mu + self.v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                            0.5*(sigma_v_a_tau**2 + sigma_v_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2)) + 1/(c_reg**2))
                
                kl_v_a_tau = -torch.log(self.tau_1) + torch.exp(self.v_tau_a_mu + 0.5*sigma_v_a_tau**2)/self.tau_1 - \
                                0.5*(self.v_tau_a_mu + 2*torch.log(sigma_v_a_tau) + 1.69315)

                kl_v_b_tau = torch.exp(0.5*sigma_v_b_tau**2 - self.v_tau_b_mu) - \
                                0.5*(2*torch.log(sigma_v_b_tau) - self.v_tau_b_mu + 1.69315)
                
                self.kl = torch.sum(kl_w) + torch.sum(kl_v) + \
                            torch.sum(kl_w_a_tau) + torch.sum(kl_v_a_tau) + torch.sum(kl_w_b_tau) + torch.sum(kl_v_b_tau)
            else:
                self.kl = torch.sum(kl_w) + torch.sum(kl_w_a_tau) + torch.sum(kl_w_b_tau)     

        return {0:self.conv2d_forward(input_dict[0], self.w, self.v), 1:Global_scale, 2:c_reg}