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
#### Spike-and-slab Gaussian batchnorm Layer
class SSGauss_VB_NormBase(torch.nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight_mu = nn.Parameter(torch.Tensor(num_features))
            self.weight_mu.prior_grad = None
            self.weight_rho = nn.Parameter(torch.Tensor(num_features))
            self.weight_rho.prior_grad = None

            self.bias_mu = nn.Parameter(torch.Tensor(num_features))
            self.bias_mu.prior_grad = None
            self.bias_rho = nn.Parameter(torch.Tensor(num_features))
            self.bias_rho.prior_grad = None

            self.theta = nn.Parameter(torch.Tensor(num_features))
            self.theta.prior_grad = None
        else:
            self.register_parameter('weight_mu', None)
            self.register_parameter('weight_rho', None)
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            self.register_parameter('theta', None)
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
            init.constant_(self.weight_rho, -6.)
            init.zeros_(self.bias_mu)
            init.constant_(self.bias_rho, -6.)
            init.constant_(self.theta, logit(torch.tensor(0.99)))

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

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class SSGauss_VB_BatchNorm(SSGauss_VB_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))
        self.register_buffer('temp', torch.as_tensor(temp))
        self.register_buffer('gamma_prior', torch.as_tensor(gamma_prior))

    def forward(self, input,  sample=False):
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
                u = torch.zeros_like(self.weight_mu).uniform_(0.0, 1.0)
                z = gumbel_softmax(self.theta, u, self.temp, hard=True)  
                
                weight_sigma = torch.log1p(torch.exp(self.weight_rho))    
                weight_epsilon = torch.zeros_like(self.weight_mu).normal_()
                weight = z * (self.weight_mu + weight_sigma * weight_epsilon)
                with torch.no_grad():
                    gamma = sigmoid(self.theta)
                    self.theta.prior_grad = gamma * (1-gamma) * (torch.log(gamma) - torch.log(self.gamma_prior) - \
                                        torch.log(1 - gamma) + torch.log(1 - self.gamma_prior) + \
                                        torch.log(self.sigma_0) - torch.log(weight_sigma) + \
                                        0.5*(weight_sigma**2 + self.weight_mu**2)/(self.sigma_0**2) - 0.5)
                    self.weight_mu.prior_grad = gamma*(self.weight_mu / self.sigma_0**2)
                    self.weight_rho.prior_grad = gamma*((weight_sigma / self.sigma_0**2 - 1 / weight_sigma) / (1 + torch.exp(-self.weight_rho)))

                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_epsilon = torch.zeros_like(self.bias_mu).normal_()
                bias = z * (self.bias_mu + bias_sigma * bias_epsilon)
                with torch.no_grad():
                    self.theta.prior_grad += gamma * (1-gamma) * (torch.log(self.sigma_0) - torch.log(bias_sigma) + \
                                        0.5*(bias_sigma**2 + self.bias_mu**2)/(self.sigma_0**2) - 0.5)
                    self.bias_mu.prior_grad = gamma*(self.bias_mu / self.sigma_0**2)
                    self.bias_rho.prior_grad = gamma*((bias_sigma / self.sigma_0**2 - 1 / bias_sigma) / (1 + torch.exp(-self.bias_rho)))

            else:
                weight = self.weight_mu
                bias = self.bias_mu             

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
        
class SSGauss_VB_BatchNorm2d(SSGauss_VB_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class SSGauss_VB_BatchNorm1d(SSGauss_VB_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

##########################################################################################################################################
#### Gaussian batchnorm Layer
class Gauss_VB_NormBase(torch.nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight_mu = nn.Parameter(torch.Tensor(num_features))
            self.weight_mu.prior_grad = None
            self.weight_rho = nn.Parameter(torch.Tensor(num_features))
            self.weight_rho.prior_grad = None

            self.bias_mu = nn.Parameter(torch.Tensor(num_features))
            self.bias_mu.prior_grad = None
            self.bias_rho = nn.Parameter(torch.Tensor(num_features))
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

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class Gauss_VB_BatchNorm(Gauss_VB_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, sigma_0 =1):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))

    def forward(self, input,  sample=False):
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
                    self.weight_mu.prior_grad = self.weight_mu / self.sigma_0**2
                    self.weight_rho.prior_grad = (weight_sigma / self.sigma_0**2 - 1 / weight_sigma) / (1 + torch.exp(-self.weight_rho)) 
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_epsilon = torch.zeros_like(self.bias_mu).normal_()
                bias = self.bias_mu + bias_sigma * bias_epsilon
                with torch.no_grad():
                    self.bias_mu.prior_grad = self.bias_mu / self.sigma_0**2
                    self.bias_rho.prior_grad = (bias_sigma / self.sigma_0**2 - 1 / bias_sigma) / (1 + torch.exp(-self.bias_rho))

            else:
                weight = self.weight_mu
                bias = self.bias_mu

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

class Gauss_VB_BatchNorm2d(Gauss_VB_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class Gauss_VB_BatchNorm1d(Gauss_VB_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
