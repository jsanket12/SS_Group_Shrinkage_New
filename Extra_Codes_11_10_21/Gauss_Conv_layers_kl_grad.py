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
#### Spike-and-slab node selection with Gaussian conv layer
class SSGauss_Node_VB_ConvNd(torch.nn.Module):

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
            self.w_mu.prior_grad = None  
            self.w_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_rho.prior_grad = None  
        else:
            self.w_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_mu.prior_grad = None  
            self.w_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_rho.prior_grad = None  
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_mu.prior_grad = None  
            self.v_rho = nn.Parameter(torch.Tensor(out_channels))
            self.v_rho.prior_grad = None  
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
        self.theta = nn.Parameter(torch.Tensor(out_channels))
        self.theta.prior_grad = None
        self.reset_parameters()

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.constant_(self.w_rho, -5)
        if self.v_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.v_mu, -bound, bound)
            init.constant_(self.v_rho, -5)
        init.constant_(self.theta, logit(torch.tensor(0.99)))

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

class SSGauss_Node_Conv2d_layer(SSGauss_Node_VB_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))
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

    def forward(self, input, sample=False):
        """
            For one Monte Carlo (MC) sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        if self.training or sample:
            u = torch.zeros_like(self.theta).uniform_(0.0, 1.0)
            z = gumbel_softmax(self.theta, u, self.temp, hard=True)
            if self.transposed:
                w_z = z.expand(
                                self.in_channels, *self.kernel_size, self.out_channels // self.groups).permute(0,3,1,2)
            else:
                w_z = z.expand( 
                                self.in_channels // self.groups, *self.kernel_size, self.out_channels).permute(3,0,1,2)
            epsilon_w = torch.zeros_like(self.w_mu).normal_()
            sigma_w = torch.log1p(torch.exp(self.w_rho))          
            self.w = w_z * (self.w_mu + sigma_w * epsilon_w)
            with torch.no_grad():
                gamma = sigmoid(self.theta)
                temp = torch.log(self.sigma_0) - torch.log(sigma_w) + \
                                        0.5*(sigma_w**2 + self.w_mu**2)/(self.sigma_0**2) - 0.5
                if self.transposed:
                    w_gamma = gamma.expand(
                                self.in_channels, *self.kernel_size, self.out_channels // self.groups).permute(0,3,1,2)
                    temp = torch.sum(temp, dim=(0,2,3)).squeeze()
                else:
                    w_gamma = gamma.expand( 
                                self.in_channels // self.groups, *self.kernel_size, self.out_channels).permute(3,0,1,2)
                    temp = torch.sum(temp, dim=(1,2,3)).squeeze()
                self.theta.prior_grad = gamma * (1-gamma) * (torch.log(gamma) - torch.log(self.gamma_prior) - \
                                        torch.log(1 - gamma) + torch.log(1 - self.gamma_prior) + \
                                        temp)
                self.w_mu.prior_grad = w_gamma*(self.w_mu / self.sigma_0**2)
                self.w_rho.prior_grad = w_gamma*((sigma_w / self.sigma_0**2 - 1 / sigma_w) / (1 + torch.exp(-self.w_rho)))
            if self.v_mu is not None:
                v_z = z
                epsilon_v = torch.zeros_like(self.v_mu).normal_()
                sigma_v = torch.log1p(torch.exp(self.v_rho))
                self.v = v_z * (self.v_mu + sigma_v * epsilon_v)
                with torch.no_grad():
                    v_gamma = gamma
                    self.theta.prior_grad += gamma * (1-gamma) * (torch.log(gamma) - torch.log(self.gamma_prior) - \
                                            torch.log(1 - gamma) + torch.log(1 - self.gamma_prior) + \
                                            torch.log(self.sigma_0) - torch.log(sigma_v) + \
                                            0.5*(sigma_v**2 + self.v_mu**2)/(self.sigma_0**2) - 0.5)
                    self.v_mu.prior_grad = v_gamma*(self.v_mu / self.sigma_0**2)
                    self.v_rho.prior_grad = v_gamma*((sigma_v / self.sigma_0**2 - 1 / sigma_v) / (1 + torch.exp(-self.v_rho)))
            else:
                self.v = None
        else:
            self.w = self.w_mu
            if self.v_mu is not None:
                self.v = self.v_mu
            else:
                self.v = None 

        return self.conv2d_forward(input, self.w, self.v)

##########################################################################################################################################
#### Spike-and-slab edge selection with Gaussian conv layer
class SSGauss_Edge_VB_ConvNd(torch.nn.Module):

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
            self.w_mu.prior_grad = None
            self.w_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_rho.prior_grad = None
            self.w_theta = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_theta.prior_grad = None
        else:
            self.w_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_mu.prior_grad = None
            self.w_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_rho.prior_grad = None
            self.w_theta = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_theta.prior_grad = None
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_mu.prior_grad = None
            self.v_rho = nn.Parameter(torch.Tensor(out_channels))
            self.v_rho.prior_grad = None
            self.v_theta = nn.Parameter(torch.Tensor(out_channels))
            self.v_theta.prior_grad = None
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
            self.register_parameter('v_theta', None)
        self.reset_parameters()

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.constant_(self.w_rho, -5)
        init.constant_(self.w_theta, logit(torch.tensor(0.99)))
        if self.v_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.v_mu, -bound, bound)
            init.constant_(self.v_rho, -5)
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

class SSGauss_Edge_Conv2d_layer(SSGauss_Edge_VB_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
    
        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))
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

    def forward(self, input, sample=False):
        """
            For one Monte Carlo (MC) sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        if self.training or sample:
            sigma_w = torch.log1p(torch.exp(self.w_rho))
            u_w = torch.zeros_like(self.w_mu).uniform_(0.0, 1.0)        
            w_z = gumbel_softmax(self.w_theta, u_w, self.temp, hard=True)        
            epsilon_w = torch.zeros_like(self.w_mu).normal_()
            self.w = w_z * (self.w_mu + sigma_w * epsilon_w)
            with torch.no_grad():
                w_gamma = sigmoid(self.w_theta)
                self.w_theta.prior_grad = w_gamma * (1-w_gamma) * (torch.log(w_gamma) - torch.log(self.gamma_prior) - \
                                        torch.log(1 - w_gamma) + torch.log(1 - self.gamma_prior) + \
                                        torch.log(self.sigma_0) - torch.log(sigma_w) + \
                                        0.5*(sigma_w**2 + self.w_mu**2)/(self.sigma_0**2) - 0.5)
                self.w_mu.prior_grad = w_gamma*(self.w_mu / self.sigma_0**2)
                self.w_rho.prior_grad = w_gamma*((sigma_w / self.sigma_0**2 - 1 / sigma_w) / (1 + torch.exp(-self.w_rho)))
            if self.v_mu is not None:
                sigma_v = torch.log1p(torch.exp(self.v_rho))
                u_v = torch.zeros_like(self.v_mu).uniform_(0.0, 1.0)
                v_z = gumbel_softmax(self.v_theta, u_v, self.temp, hard=True)
                epsilon_v = torch.zeros_like(self.v_mu).normal_()
                self.v = v_z * (self.v_mu + sigma_v * epsilon_v)
                with torch.no_grad():
                    v_gamma = sigmoid(self.v_theta)
                    self.v_theta.prior_grad = v_gamma * (1-v_gamma) * (torch.log(v_gamma) - torch.log(self.gamma_prior) - \
                                            torch.log(1 - v_gamma) + torch.log(1 - self.gamma_prior) + \
                                            torch.log(self.sigma_0) - torch.log(sigma_v) + \
                                            0.5*(sigma_v**2 + self.v_mu**2)/(self.sigma_0**2) - 0.5)
                    self.v_mu.prior_grad = v_gamma*(self.v_mu / self.sigma_0**2)
                    self.v_rho.prior_grad = v_gamma*((sigma_v / self.sigma_0**2 - 1 / sigma_v) / (1 + torch.exp(-self.v_rho)))
            else:
                self.v = None
        else:
            self.w = self.w_mu
            if self.v_mu is not None:
                self.v = self.v_mu
            else:
                self.v = None    

        return self.conv2d_forward(input, self.w, self.v)

##########################################################################################################################################
#### Gaussian without spike-and-slab conv layer
class Gauss_VB_ConvNd(torch.nn.Module):

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
            self.w_mu.prior_grad = None
            self.w_rho = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.w_rho.prior_grad = None
        else:
            self.w_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_mu.prior_grad = None
            self.w_rho = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.w_rho.prior_grad = None
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(out_channels))
            self.v_mu.prior_grad = None
            self.v_rho = nn.Parameter(torch.Tensor(out_channels))
            self.v_rho.prior_grad = None
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
        self.reset_parameters()

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.constant_(self.w_rho, -5)
        if self.v_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.v_mu, -bound, bound)
            init.constant_(self.v_rho, -5)

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

class Gauss_Conv2d_layer(Gauss_VB_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', sigma_0 =1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        
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

    def forward(self, input, sample=False):
        """
            For one Monte Carlo (MC) sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        if self.training or sample:
            sigma_w = torch.log1p(torch.exp(self.w_rho))
            epsilon_w = torch.zeros_like(self.w_mu).normal_()
            self.w = self.w_mu + sigma_w * epsilon_w
            with torch.no_grad():
                self.w_mu.prior_grad = self.w_mu / self.sigma_0**2
                self.w_rho.prior_grad = (sigma_w / self.sigma_0**2 - 1 / sigma_w) / (1 + torch.exp(-self.w_rho)) 
            if self.v_mu is not None:
                sigma_v = torch.log1p(torch.exp(self.v_rho))
                epsilon_v = torch.zeros_like(self.v_mu).normal_()
                self.v = self.v_mu + sigma_v * epsilon_v
                with torch.no_grad():
                    self.v_mu.prior_grad = self.v_mu / self.sigma_0**2
                    self.v_rho.prior_grad = (sigma_v / self.sigma_0**2 - 1 / sigma_v) / (1 + torch.exp(-self.v_rho))
            else:
                self.v = None
        else:
            self.w = self.w_mu
            if self.v_mu is not None:
                self.v = self.v_mu
            else:
                self.v = None
                
        return self.conv2d_forward(input, self.w, self.v)