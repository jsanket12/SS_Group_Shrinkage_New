import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.modules.utils import _single, _pair, _triple

from collections import OrderedDict, namedtuple

from itertools import islice
import operator
import numpy as np

#
# class Gaussian(object):
#     def __init__(self, mu, rho):
#         super().__init__()
#         self.mu = mu
#         self.rho = rho
#         # self.normal = torch.distributions.Normal(0, 1)
#         # self.normal = torch.zeros_like(mu)
#
#     @property
#     def sigma(self):
#         return torch.log1p(torch.exp(self.rho))
#
#     def sample(self):
#         # epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
#         epsilon = torch.zeros_like(self.mu).normal_()
#         # epsilon = self.normal.normal_()
#         return self.mu + self.sigma * epsilon
#
#     def log_prob(self, input):
#         return (-math.log(math.sqrt(2 * math.pi))
#                 - torch.log(self.sigma)
#                 - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()
#
#
#
# class ScaleMixtureGaussian(object):
#     def __init__(self, pi, sigma1, sigma2):
#         super().__init__()
#         self.pi = pi
#         self.sigma1 = sigma1
#         self.sigma2 = sigma2
#         self.gaussian1 = torch.distributions.Normal(0, sigma1)
#         self.gaussian2 = torch.distributions.Normal(0, sigma2)
#
#     def log_prob(self, input):
#         prob1 = torch.exp(self.gaussian1.log_prob(input))
#         prob2 = torch.exp(self.gaussian2.log_prob(input))
#         return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()

class VB_Linear(torch.nn.Module):

    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features,  bias=True, lambda_n = 0.01, sigma_0 = 0.00001, sigma_1 = 0.01):
        super(VB_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_mu.prior_grad = None
        self.weight_rho = Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho.prior_grad = None
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_mu.prior_grad = None
            self.bias_rho = Parameter(torch.Tensor(out_features))
            self.bias_rho.prior_grad = None
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        self.reset_parameters()
        self.c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        self.c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        self.threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma_1 / sigma_0)) / (
                0.5 / sigma_0 - 0.5 / sigma_1))
        self.lambda_n = lambda_n
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        init.constant_(self.weight_rho, -5)
        if self.bias_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_mu, -bound, bound)
            init.constant_(self.bias_rho, -5)
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias_mu is not None
        )

    def set_prior(self, lambda_n, sigma_0, sigma_1):
        self.c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        self.c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        self.threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma_1 / sigma_0)) / (
                0.5 / sigma_0 - 0.5 / sigma_1))
        self.lambda_n = lambda_n
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight_epsilon = torch.zeros_like(self.weight_mu).normal_()
            weight = self.weight_mu + weight_sigma * weight_epsilon
            with torch.no_grad():
                temp = weight.pow(2).mul(self.c2).add(self.c1).exp().add(1).pow(-1)
                temp = weight.div(-self.sigma_0).mul(temp) + weight.div(-self.sigma_1).mul(1 - temp)
                self.weight_mu.prior_grad = - temp
                self.weight_rho.prior_grad = - temp * weight_epsilon / (1 + torch.exp(-self.weight_rho)) - 1 / (
                    weight_sigma) * (1/(1 + torch.exp(-self.weight_rho)) )
            if self.bias_mu is not None:
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_epsilon = torch.zeros_like(self.bias_mu).normal_()
                bias = self.bias_mu + bias_sigma * bias_epsilon
                with torch.no_grad():
                    temp = bias.pow(2).mul(self.c2).add(self.c1).exp().add(1).pow(-1)
                    temp = bias.div(-self.sigma_0).mul(temp) + bias.div(-self.sigma_1).mul(1 - temp)
                    self.bias_mu.prior_grad = - temp
                    self.bias_rho.prior_grad = - temp * bias_epsilon / (1 + torch.exp(-self.bias_rho)) - 1 / (
                        bias_sigma) * (1 / (1 + torch.exp(-self.bias_rho)))
            else:
                bias = None
        else:
            # weight = self.weight_mu
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight_epsilon = torch.zeros_like(self.weight_mu).normal_()
            weight = self.weight_mu + weight_sigma * weight_epsilon
            if self.bias_mu is not None:
                # bias = self.bias_mu
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_epsilon = torch.zeros_like(self.bias_mu).normal_()
                bias = self.bias_mu + bias_sigma * bias_epsilon
            else:
                bias = None
        return F.linear(input, weight, bias)



class _VB_ConvNd(torch.nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_VB_ConvNd, self).__init__()
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
            self.weight_mu = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.weight_mu.prior_grad = None
            self.weight_rho = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.weight_rho.prior_grad = None
        else:
            self.weight_mu = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.weight_mu.prior_grad = None
            self.weight_rho = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.weight_rho.prior_grad = None
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_mu.prior_grad = None
            self.bias_rho = Parameter(torch.Tensor(out_channels))
            self.bias_rho.prior_grad = None
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        self.reset_parameters()

        self.input_size = None


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        init.constant_(self.weight_rho, -5)
        if self.bias_mu is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_mu, -bound, bound)
            init.constant_(self.bias_rho, -5)

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
        super(_VB_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class VB_Conv2d(_VB_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', lambda_n = 0.01, sigma_0 = 0.00001, sigma_1 = 0.01):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(VB_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        self.c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        self.threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma_1 / sigma_0)) / (
                0.5 / sigma_0 - 0.5 / sigma_1))
        self.lambda_n = lambda_n
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1

    def set_prior(self, lambda_n, sigma_0, sigma_1):
        self.c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        self.c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        self.threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma_1 / sigma_0)) / (
                0.5 / sigma_0 - 0.5 / sigma_1))
        self.lambda_n = lambda_n
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1

    def conv2d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                weight, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def forward(self, input, sample=False, calculate_log_probs=False):
        self.input_size = input.size()
        if self.training or sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight_epsilon = torch.zeros_like(self.weight_mu).normal_()
            weight = self.weight_mu + weight_sigma * weight_epsilon
            with torch.no_grad():
                temp = weight.pow(2).mul(self.c2).add(self.c1).exp().add(1).pow(-1)
                temp = weight.div(-self.sigma_0).mul(temp) + weight.div(-self.sigma_1).mul(1 - temp)
                self.weight_mu.prior_grad = - temp
                self.weight_rho.prior_grad = - temp * weight_epsilon / (1 + torch.exp(-self.weight_rho)) - 1 / (
                    weight_sigma) * (1/(1 + torch.exp(-self.weight_rho)) )
            if self.bias_mu is not None:
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_epsilon = torch.zeros_like(self.bias_mu).normal_()
                bias = self.bias_mu + bias_sigma * bias_epsilon
                with torch.no_grad():
                    temp = bias.pow(2).mul(self.c2).add(self.c1).exp().add(1).pow(-1)
                    temp = bias.div(-self.sigma_0).mul(temp) + bias.div(-self.sigma_1).mul(1 - temp)
                    self.bias_mu.prior_grad = - temp
                    self.bias_rho.prior_grad = - temp * bias_epsilon / (1 + torch.exp(-self.bias_rho)) - 1 / (
                        bias_sigma) * (1 / (1 + torch.exp(-self.bias_rho)))
            else:
                bias = None
        else:
            # weight = self.weight_mu
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight_epsilon = torch.zeros_like(self.weight_mu).normal_()
            weight = self.weight_mu + weight_sigma * weight_epsilon
            if self.bias_mu is not None:
                # bias = self.bias_mu
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_epsilon = torch.zeros_like(self.bias_mu).normal_()
                bias = self.bias_mu + bias_sigma * bias_epsilon
            else:
                bias = None
        return self.conv2d_forward(input, weight, bias)



class _VB_NormBase(torch.nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_VB_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats


        if self.affine:
            self.weight_mu = Parameter(torch.Tensor(num_features))
            self.weight_mu.prior_grad = None
            self.weight_rho = Parameter(torch.Tensor(num_features))
            self.weight_rho.prior_grad = None

            self.bias_mu = Parameter(torch.Tensor(num_features))
            self.bias_mu.prior_grad = None
            self.bias_rho = Parameter(torch.Tensor(num_features))
            self.bias_rho.prior_grad = None

        else:
            self.register_parameter('weight_mu', None)
            self.register_parameter('weight_rho', None)
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight_mu)
            init.constant_(self.weight_rho, -5)
            init.zeros_(self.bias_mu)
            init.constant_(self.bias_rho, -5)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_VB_NormBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class _VB_BatchNorm(_VB_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, lambda_n = 0.01, sigma_0 = 0.00001, sigma_1 = 0.01):
        super(_VB_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        self.c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        self.threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma_1 / sigma_0)) / (
                0.5 / sigma_0 - 0.5 / sigma_1))
        self.lambda_n = lambda_n
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
    def set_prior(self, lambda_n, sigma_0, sigma_1):
        self.c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        self.c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        self.threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma_1 / sigma_0)) / (
                0.5 / sigma_0 - 0.5 / sigma_1))
        self.lambda_n = lambda_n
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
    def forward(self, input,  sample=False, calculate_log_probs=False):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.affine:
            # return F.batch_norm(
            # input, self.running_mean, self.running_var, self.weight.mul(self.weight_mask), self.bias.mul(self.bias_mask),
            # self.training or not self.track_running_stats,
            # exponential_average_factor, self.eps)

            if self.training or sample:
                weight_sigma = torch.log1p(torch.exp(self.weight_rho))
                weight_epsilon = torch.zeros_like(self.weight_mu).normal_()
                weight = self.weight_mu + weight_sigma * weight_epsilon
                with torch.no_grad():
                    temp = weight.pow(2).mul(self.c2).add(self.c1).exp().add(1).pow(-1)
                    temp = weight.div(-self.sigma_0).mul(temp) + weight.div(-self.sigma_1).mul(1 - temp)
                    self.weight_mu.prior_grad = - temp
                    self.weight_rho.prior_grad = - temp * weight_epsilon / (1 + torch.exp(-self.weight_rho)) - 1 / (
                        weight_sigma) * (1 / (1 + torch.exp(-self.weight_rho)))
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_epsilon = torch.zeros_like(self.bias_mu).normal_()
                bias = self.bias_mu + bias_sigma * bias_epsilon
                with torch.no_grad():
                    temp = bias.pow(2).mul(self.c2).add(self.c1).exp().add(1).pow(-1)
                    temp = bias.div(-self.sigma_0).mul(temp) + bias.div(-self.sigma_1).mul(1 - temp)
                    self.bias_mu.prior_grad = - temp
                    self.bias_rho.prior_grad = - temp * bias_epsilon / (1 + torch.exp(-self.bias_rho)) - 1 / (
                        bias_sigma) * (1 / (1 + torch.exp(-self.bias_rho)))
            else:
                # weight = self.weight_mu
                weight_sigma = torch.log1p(torch.exp(self.weight_rho))
                weight_epsilon = torch.zeros_like(self.weight_mu).normal_()
                weight = self.weight_mu + weight_sigma * weight_epsilon
                # bias = self.bias_mu
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_epsilon = torch.zeros_like(self.bias_mu).normal_()
                bias = self.bias_mu + bias_sigma * bias_epsilon

            return F.batch_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        else:
            weight = None
            bias = None
            return F.batch_norm(
                input, self.running_mean, self.running_var, weight, bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)

class VB_BatchNorm2d(_VB_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class VB_BatchNorm1d(_VB_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))