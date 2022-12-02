import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
torch.set_default_dtype(torch.float32)
# pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2).type(torch.float)

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
class SS_GHS_Node_layer(nn.Module):

    __constants__ = ['bias', 'input_dim', 'output_dim']

    def __init__(self, input_dim, output_dim, bias=True, clip_std=None,
                    temp = 0.5, gamma_prior = 0.0001, sigma_0 = 1, tau_1 = 1):
        super().__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer('temp', torch.as_tensor(temp))
        self.register_buffer('tau_1', torch.as_tensor(tau_1))
        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))
        self.register_buffer('gamma_prior', torch.as_tensor(gamma_prior))
        self.clip_std = clip_std

        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.theta = nn.Parameter(torch.Tensor(output_dim))  
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(output_dim))
            self.v_rho = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
        self.tau_a_mu = nn.Parameter(torch.Tensor(output_dim))
        self.tau_b_mu = nn.Parameter(torch.Tensor(output_dim))
        self.tau_a_rho = nn.Parameter(torch.Tensor(output_dim))
        self.tau_b_rho = nn.Parameter(torch.Tensor(output_dim))
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
        init.constant_(self.theta, logit(torch.tensor(0.99)))
        if self.v_mu is not None:
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
        return 'input_dim={}, output_dim={}, bias={}'.format(
            self.input_dim, self.output_dim, self.v_mu is not None
        )

    def forward(self, input_dict):
        """
            For one Monte Carlo (MC) sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        Global_scale = input_dict[1]
        u = torch.zeros_like(self.theta).uniform_(0.0, 1.0)
        z = gumbel_softmax(self.theta, u, self.temp, hard=True)
        w_z = z.expand(self.input_dim, self.output_dim)
        epsilon_w = torch.zeros_like(self.w_mu).normal_()            
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        
        sigma_a_tau = torch.log1p(torch.exp(self.tau_a_rho))
        sigma_b_tau = torch.log1p(torch.exp(self.tau_b_rho))
        sigma_tau = 0.5*torch.sqrt(sigma_a_tau**2 + sigma_b_tau**2)
        epsilon_tau = torch.zeros_like(self.tau_a_mu).normal_()
        GHS_scale = Global_scale * \
                        torch.exp(0.5*(self.tau_a_mu+self.tau_b_mu) + sigma_tau * epsilon_tau)
        self.w = w_z * GHS_scale.expand(self.input_dim, self.output_dim) *(self.w_mu + sigma_w * epsilon_w)

        if self.v_mu is not None:
            epsilon_v = torch.zeros_like(self.v_mu).normal_()
            sigma_v = torch.log1p(torch.exp(self.v_rho))
            self.v = z * GHS_scale * (self.v_mu + sigma_v * epsilon_v)
        else:
            self.v = None 
        
        if self.training:
            gamma = sigmoid(self.theta)

            kl_gamma = gamma * (torch.log(gamma) - torch.log(self.gamma_prior)) + \
                        (1 - gamma) * (torch.log(1 - gamma) - torch.log(1 - self.gamma_prior)) 
        
            kl_w = torch.log(self.sigma_0) - torch.log(sigma_w) - 0.5 + \
                        0.5*((sigma_w**2+self.w_mu**2)/self.sigma_0**2)
            
            kl_a_tau = -torch.log(self.tau_1) + torch.exp(self.tau_a_mu + 0.5*sigma_a_tau**2)/self.tau_1 - \
                            0.5*(self.tau_a_mu + 2*torch.log(sigma_a_tau) + 1.69315)

            kl_b_tau = torch.exp(0.5*sigma_b_tau**2 - self.tau_b_mu) - \
                            0.5*(2*torch.log(sigma_b_tau) - self.tau_b_mu + 1.69315)

            if self.v_mu is not None:
                kl_v = torch.log(self.sigma_0) - torch.log(sigma_v) - 0.5 + \
                        0.5*((sigma_v**2+self.v_mu**2)/self.sigma_0**2)
                
                self.kl = torch.sum(kl_gamma) + \
                            torch.sum(kl_w) + torch.sum(kl_v) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)
            else:
                self.kl = torch.sum(kl_gamma) + \
                            torch.sum(kl_w) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)              

        return {0:F.linear(input_dict[0], self.w.T, self.v), 1:Global_scale}

##########################################################################################################################################
#### Group Horseshoe without spike-and-slab Layer
class GHS_layer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, clip_std=None, sigma_0=1, tau_1 =1):
        super().__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer('sigma_0', torch.as_tensor(sigma_0))
        self.register_buffer('tau_1', torch.as_tensor(tau_1))
        self.clip_std = clip_std

        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if bias:
            self.v_mu = nn.Parameter(torch.Tensor(output_dim))
            self.v_rho = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('v_mu', None)
            self.register_parameter('v_rho', None)
        self.tau_a_mu = nn.Parameter(torch.Tensor(output_dim))
        self.tau_b_mu = nn.Parameter(torch.Tensor(output_dim))
        self.tau_a_rho = nn.Parameter(torch.Tensor(output_dim))
        self.tau_b_rho = nn.Parameter(torch.Tensor(output_dim))
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

    def clip_standard_dev(self):
        if self.clip_std:
            self.w_rho.data.clamp_(max=math.log(math.expm1(self.clip_std)))
            if self.v_mu is not None:
                self.v_rho.data.clamp_(max=math.log(math.expm1(self.clip_std)))

    def extra_repr(self):
        return 'input_dim={}, output_dim={}, bias={}'.format(
            self.input_dim, self.output_dim, self.v_mu is not None
        )

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
        self.w = GHS_scale.expand(self.input_dim, self.output_dim) *(self.w_mu + sigma_w * epsilon_w)
        if self.v_mu is not None:
            epsilon_v = torch.zeros_like(self.v_mu).normal_()
            sigma_v = torch.log1p(torch.exp(self.v_rho))
            self.v = GHS_scale * (self.v_mu + sigma_v * epsilon_v)
        else:
            self.v = None

        if self.training:
            kl_w = torch.log(self.sigma_0) - torch.log(sigma_w) - 0.5 + \
                        0.5*((sigma_w**2+self.w_mu**2)/self.sigma_0**2)
            
            kl_a_tau = -torch.log(self.tau_1) + torch.exp(self.tau_a_mu + 0.5*sigma_a_tau**2)/self.tau_1 - \
                            0.5*(self.tau_a_mu + 2*torch.log(sigma_a_tau) + 1.69315)

            kl_b_tau = torch.exp(0.5*sigma_b_tau**2 - self.tau_b_mu) - \
                            0.5*(2*torch.log(sigma_b_tau) - self.tau_b_mu + 1.69315)

            if self.v_mu is not None:
                kl_v = torch.log(self.sigma_0) - torch.log(sigma_v) - 0.5 + \
                        0.5*((sigma_v**2+self.v_mu**2)/self.sigma_0**2)
                
                self.kl = torch.sum(kl_w) + torch.sum(kl_v) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau) 
            else:
                self.kl = torch.sum(kl_w) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)                

        return {0:F.linear(input_dict[0], self.w.T, self.v), 1:Global_scale}