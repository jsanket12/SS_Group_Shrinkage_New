import torch
import torch.nn as nn
import torch.nn.init as init
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
#### Spike-and-slab simultaneous node and edge selection with group horseshoe Layer
class SS_GHS_Node_and_Edge_layer(nn.Module):
    def __init__(self, input_dim, output_dim, rho_prior, temp, gamma_prior, gamma_prior_star, 
                sig_a_mu, sig_a_rho, sig_b_mu, sig_b_rho, lambda0=0.99):
        super().__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sig_a_mu = sig_a_mu 
        self.sig_a_rho = sig_a_rho
        self.sig_b_mu = sig_b_mu 
        self.sig_b_rho = sig_b_rho
        self.register_buffer('temp', torch.as_tensor(temp))
        self.register_buffer('rho_prior', torch.as_tensor(rho_prior))
        self.register_buffer('gamma_prior', torch.as_tensor(gamma_prior))
        self.register_buffer('gamma_prior_star', torch.as_tensor(gamma_prior_star))

        # initialize mu and rho parameters for hidden layer's weights, and theta = logit(gamma)
        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim))
        init.constant_(self.w_rho, -5)   
        self.w_theta = nn.Parameter(torch.Tensor(input_dim, output_dim))
        init.constant_(self.w_theta, logit(torch.tensor(lambda0)))

        # initialize mu and rho parameters for hidden layer's biases
        self.v_mu = nn.Parameter(torch.Tensor(output_dim))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.v_mu, -bound, bound)
        self.v_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.v_rho, -5)
        self.v_theta = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.v_theta, logit(torch.tensor(lambda0)))

        self.tau_a_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.tau_b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.tau_a_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.tau_a_rho, -5)
        self.tau_b_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.tau_b_rho, -5)

        self.theta_star = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.theta_star, logit(torch.tensor(lambda0)))

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

        # initialize kl for the hidden layer
        self.kl = 0

    def forward(self, X):
        """
            For one Monte Carlo (MC) sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        # sample weights and biases
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        sigma_v = torch.log1p(torch.exp(self.v_rho))
        sigma_a_tau = torch.log1p(torch.exp(self.tau_a_rho))
        sigma_b_tau = torch.log1p(torch.exp(self.tau_b_rho))
        sigma_prior = torch.log1p(torch.exp(self.rho_prior)) 
        sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
        sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho))
        
        u = torch.zeros_like(self.theta_star).uniform_(0.0, 1.0)
        z_star = gumbel_softmax(self.theta_star, u, self.temp, hard=True)
        z_star_w = z_star.expand(self.input_dim, self.output_dim)
        z_star_v = z_star
        
        u_w = torch.zeros_like(self.w_theta).uniform_(0.0, 1.0)        
        w_z = z_star_w * gumbel_softmax(self.w_theta, u_w, self.temp, hard=True)  
        u_v = torch.zeros_like(self.v_theta).uniform_(0.0, 1.0)
        v_z = z_star_v * gumbel_softmax(self.v_theta, u_v, self.temp, hard=True)
        
        epsilon_w = torch.zeros_like(self.w_mu).normal_()
        epsilon_v = torch.zeros_like(self.v_mu).normal_()

        self.w = w_z * (self.w_mu + sigma_w * epsilon_w)
        self.v = v_z * (self.v_mu + sigma_v * epsilon_v)
        output = torch.mm(X, self.w) + self.v.expand(X.size()[0], self.output_dim)

        # record KL at sampled weight and bias with sampled inclusion probabilities
        gamma_star = sigmoid(self.theta_star)
        w_gamma_star = gamma_star.expand(self.input_dim, self.output_dim)
        v_gamma_star = gamma_star
        w_gamma = sigmoid(self.w_theta)
        v_gamma = sigmoid(self.v_theta)
        w_tau_a_mu = self.tau_a_mu.expand(self.input_dim, self.output_dim)
        v_tau_a_mu = self.tau_a_mu
        w_tau_b_mu = self.tau_b_mu.expand(self.input_dim, self.output_dim)
        v_tau_b_mu = self.tau_b_mu
        sigma_w_a_tau = sigma_a_tau.expand(self.input_dim, self.output_dim)
        sigma_v_a_tau = sigma_a_tau
        sigma_w_b_tau = sigma_b_tau.expand(self.input_dim, self.output_dim)
        sigma_v_b_tau = sigma_b_tau

        kl_gamma_star = gamma_star * (torch.log(gamma_star) - torch.log(self.gamma_prior_star)) + \
               (1 - gamma_star) * (torch.log(1 - gamma_star) - torch.log(1 - self.gamma_prior_star))

        kl_w_gamma = w_gamma * (torch.log(w_gamma) - torch.log(self.gamma_prior)) + \
               (1 - w_gamma) * (torch.log(1 - w_gamma) - torch.log(1 - self.gamma_prior)) 

        kl_v_gamma = v_gamma * (torch.log(v_gamma) - torch.log(self.gamma_prior)) + \
               (1 - v_gamma) * (torch.log(1 - v_gamma) - torch.log(1 - self.gamma_prior)) 
        
        kl_w = w_gamma_star * w_gamma * (torch.log(sigma_prior) - torch.log(sigma_w) - 0.5 + \
                    0.5*(w_tau_a_mu + w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                    0.5*((sigma_w**2+self.w_mu**2)/sigma_prior**2)* \
                    torch.exp(-(w_tau_a_mu + w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                        0.5*(sigma_w_a_tau**2 + sigma_w_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2)))
        
        kl_v = v_gamma_star * v_gamma * (torch.log(sigma_prior) - torch.log(sigma_v) - 0.5 + \
                        0.5*(v_tau_a_mu + v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                        0.5*((sigma_v**2+self.v_mu**2)/sigma_prior**2)* \
                        torch.exp(-(v_tau_a_mu + v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                            0.5*(sigma_v_a_tau**2 + sigma_v_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2)))
        
        kl_a_tau = torch.exp(self.tau_a_mu + 0.5*sigma_a_tau**2) - \
                        0.5*(self.tau_a_mu + 2*torch.log(sigma_a_tau) + 1.69315)

        kl_b_tau = torch.exp(0.5*sigma_b_tau**2 - self.tau_b_mu) - \
                        0.5*(2*torch.log(sigma_b_tau) - self.tau_b_mu + 1.69315)

        self.kl = torch.sum(kl_w_gamma) + torch.sum(kl_v_gamma) + torch.sum(kl_gamma_star)  + \
                    torch.sum(kl_w) + torch.sum(kl_v) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)       

        return output

##########################################################################################################################################
#### Spike-and-slab node selection with group horseshoe Layer
class SS_GHS_Node_layer(nn.Module):
    def __init__(self, input_dim, output_dim, rho_prior, temp, gamma_prior, 
                sig_a_mu, sig_a_rho, sig_b_mu, sig_b_rho, lambda0=0.99):
        super().__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sig_a_mu = sig_a_mu 
        self.sig_a_rho = sig_a_rho
        self.sig_b_mu = sig_b_mu 
        self.sig_b_rho = sig_b_rho
        self.register_buffer('temp', torch.as_tensor(temp))
        self.register_buffer('rho_prior', torch.as_tensor(rho_prior))
        self.register_buffer('gamma_prior', torch.as_tensor(gamma_prior))

        # initialize mu and rho parameters for hidden layer's weights, and theta = logit(gamma)
        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim))
        init.constant_(self.w_rho, -5)
        
        # initialize mu and rho parameters for hidden layer's biases
        self.v_mu = nn.Parameter(torch.Tensor(output_dim))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.v_mu, -bound, bound)
        self.v_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.v_rho, -5)        

        self.tau_a_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.tau_b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.tau_a_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.tau_a_rho, -5)
        self.tau_b_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.tau_b_rho, -5)

        self.theta = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.theta, logit(torch.tensor(lambda0)))

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

        # initialize kl for the hidden layer
        self.kl = 0

    def forward(self, X):
        """
            For one Monte Carlo (MC) sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        # sample weights and biases
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        sigma_v = torch.log1p(torch.exp(self.v_rho))
        sigma_a_tau = torch.log1p(torch.exp(self.tau_a_rho))
        sigma_b_tau = torch.log1p(torch.exp(self.tau_b_rho))
        sigma_prior = torch.log1p(torch.exp(self.rho_prior)) 
        sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
        sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho))
        
        u = torch.zeros_like(self.theta).uniform_(0.0, 1.0)
        z = gumbel_softmax(self.theta, u, self.temp, hard=True)
        w_z = z.expand(self.input_dim, self.output_dim)
        v_z = z
        
        epsilon_w = torch.zeros_like(self.w_mu).normal_()
        epsilon_v = torch.zeros_like(self.v_mu).normal_()
        
        self.w = w_z * (self.w_mu + sigma_w * epsilon_w)
        self.v = v_z * (self.v_mu + sigma_v * epsilon_v)
        output = torch.mm(X, self.w) + self.v.expand(X.size()[0], self.output_dim)

        # record KL at sampled weight and bias with sampled inclusion probabilities
        gamma = sigmoid(self.theta)
        w_gamma = gamma.expand(self.input_dim, self.output_dim)
        v_gamma = gamma
        w_tau_a_mu = self.tau_a_mu.expand(self.input_dim, self.output_dim)
        v_tau_a_mu = self.tau_a_mu
        w_tau_b_mu = self.tau_b_mu.expand(self.input_dim, self.output_dim)
        v_tau_b_mu = self.tau_b_mu
        sigma_w_a_tau = sigma_a_tau.expand(self.input_dim, self.output_dim)
        sigma_v_a_tau = sigma_a_tau
        sigma_w_b_tau = sigma_b_tau.expand(self.input_dim, self.output_dim)
        sigma_v_b_tau = sigma_b_tau

        kl_gamma = gamma * (torch.log(gamma) - torch.log(self.gamma_prior)) + \
               (1 - gamma) * (torch.log(1 - gamma) - torch.log(1 - self.gamma_prior)) 
        
        kl_w = w_gamma * (torch.log(sigma_prior) - torch.log(sigma_w) - 0.5 + \
                    0.5*(w_tau_a_mu + w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                    0.5*((sigma_w**2+self.w_mu**2)/sigma_prior**2)* \
                    torch.exp(-(w_tau_a_mu + w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                        0.5*(sigma_w_a_tau**2 + sigma_w_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2)))
        
        kl_v = v_gamma * (torch.log(sigma_prior) - torch.log(sigma_v) - 0.5 + \
                        0.5*(v_tau_a_mu + v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                        0.5*((sigma_v**2+self.v_mu**2)/sigma_prior**2)* \
                        torch.exp(-(v_tau_a_mu + v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                            0.5*(sigma_v_a_tau**2 + sigma_v_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2)))
        
        kl_a_tau = torch.exp(self.tau_a_mu + 0.5*sigma_a_tau**2) - \
                        0.5*(self.tau_a_mu + 2*torch.log(sigma_a_tau) + 1.69315)

        kl_b_tau = torch.exp(0.5*sigma_b_tau**2 - self.tau_b_mu) - \
                        0.5*(2*torch.log(sigma_b_tau) - self.tau_b_mu + 1.69315)

        self.kl = torch.sum(kl_gamma) + \
                    torch.sum(kl_w) + torch.sum(kl_v) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)
        return output

##########################################################################################################################################
#### Spike-and-slab edge selection with group horseshoe Layer
class SS_GHS_Edge_layer(nn.Module):
    def __init__(self, input_dim, output_dim, rho_prior, temp, gamma_prior, 
                sig_a_mu, sig_a_rho, sig_b_mu, sig_b_rho, lambda0=0.99):
        super().__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sig_a_mu = sig_a_mu 
        self.sig_a_rho = sig_a_rho
        self.sig_b_mu = sig_b_mu 
        self.sig_b_rho = sig_b_rho
        self.register_buffer('temp', torch.as_tensor(temp))
        self.register_buffer('rho_prior', torch.as_tensor(rho_prior))
        self.register_buffer('gamma_prior', torch.as_tensor(gamma_prior))

        # initialize mu and rho parameters for hidden layer's weights, and theta = logit(gamma)
        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim))
        init.constant_(self.w_rho, -5)
        self.w_theta = nn.Parameter(torch.Tensor(input_dim, output_dim))
        init.constant_(self.w_theta, logit(torch.tensor(lambda0)))
        
        # initialize mu and rho parameters for hidden layer's biases
        self.v_mu = nn.Parameter(torch.Tensor(output_dim))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.v_mu, -bound, bound)
        self.v_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.v_rho, -5)
        self.v_theta = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.v_theta, logit(torch.tensor(lambda0)))

        self.tau_a_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.tau_b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.tau_a_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.tau_a_rho, -5)
        self.tau_b_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.tau_b_rho, -5)

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

        # initialize kl for the hidden layer
        self.kl = 0

    def forward(self, X):
        """
            For one Monte Carlo (MC) sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        # sample weights and biases
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        sigma_v = torch.log1p(torch.exp(self.v_rho))
        sigma_a_tau = torch.log1p(torch.exp(self.tau_a_rho))
        sigma_b_tau = torch.log1p(torch.exp(self.tau_b_rho))
        sigma_prior = torch.log1p(torch.exp(self.rho_prior)) 
        sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
        sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho))
        
        u_w = torch.zeros_like(self.w_mu).uniform_(0.0, 1.0)        
        w_z = gumbel_softmax(self.w_theta, u_w, self.temp, hard=True)   
        u_v = torch.zeros_like(self.v_mu).uniform_(0.0, 1.0)
        v_z = gumbel_softmax(self.v_theta, u_v, self.temp, hard=True)

        epsilon_w = torch.zeros_like(self.w_mu).normal_()
        epsilon_v = torch.zeros_like(self.v_mu).normal_()
        
        self.w = w_z * (self.w_mu + sigma_w * epsilon_w)
        self.v = v_z * (self.v_mu + sigma_v * epsilon_v)
        output = torch.mm(X, self.w) + self.v.expand(X.size()[0], self.output_dim)

        # record KL at sampled weight and bias with sampled inclusion probabilities
        w_gamma = sigmoid(self.w_theta)
        v_gamma = sigmoid(self.v_theta)
        w_tau_a_mu = self.tau_a_mu.expand(self.input_dim, self.output_dim)
        v_tau_a_mu = self.tau_a_mu
        w_tau_b_mu = self.tau_b_mu.expand(self.input_dim, self.output_dim)
        v_tau_b_mu = self.tau_b_mu
        sigma_w_a_tau = sigma_a_tau.expand(self.input_dim, self.output_dim)
        sigma_v_a_tau = sigma_a_tau
        sigma_w_b_tau = sigma_b_tau.expand(self.input_dim, self.output_dim)
        sigma_v_b_tau = sigma_b_tau

        kl_w_gamma = w_gamma * (torch.log(w_gamma) - torch.log(self.gamma_prior)) + \
               (1 - w_gamma) * (torch.log(1 - w_gamma) - torch.log(1 - self.gamma_prior)) 

        kl_v_gamma = v_gamma * (torch.log(v_gamma) - torch.log(self.gamma_prior)) + \
               (1 - v_gamma) * (torch.log(1 - v_gamma) - torch.log(1 - self.gamma_prior)) 
        
        kl_w = w_gamma * (torch.log(sigma_prior) - torch.log(sigma_w) - 0.5 + \
                    0.5*(w_tau_a_mu + w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                    0.5*((sigma_w**2+self.w_mu**2)/sigma_prior**2)* \
                    torch.exp(-(w_tau_a_mu + w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                        0.5*(sigma_w_a_tau**2 + sigma_w_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2)))
        
        kl_v = v_gamma * (torch.log(sigma_prior) - torch.log(sigma_v) - 0.5 + \
                        0.5*(v_tau_a_mu + v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                        0.5*((sigma_v**2+self.v_mu**2)/sigma_prior**2)* \
                        torch.exp(-(v_tau_a_mu + v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                            0.5*(sigma_v_a_tau**2 + sigma_v_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2)))
        
        kl_a_tau = torch.exp(self.tau_a_mu + 0.5*sigma_a_tau**2) - \
                        0.5*(self.tau_a_mu + 2*torch.log(sigma_a_tau) + 1.69315)

        kl_b_tau = torch.exp(0.5*sigma_b_tau**2 - self.tau_b_mu) - \
                        0.5*(2*torch.log(sigma_b_tau) - self.tau_b_mu + 1.69315)

        self.kl = torch.sum(kl_w_gamma) + torch.sum(kl_v_gamma) + \
                    torch.sum(kl_w) + torch.sum(kl_v) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)             

        return output

##########################################################################################################################################
#### Group Horseshoe without spike-and-slab Layer
class GHS_layer(nn.Module):
    def __init__(self, input_dim, output_dim, rho_prior, sig_a_mu, sig_a_rho, sig_b_mu, sig_b_rho):
        super().__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sig_a_mu = sig_a_mu 
        self.sig_a_rho = sig_a_rho
        self.sig_b_mu = sig_b_mu 
        self.sig_b_rho = sig_b_rho
        self.register_buffer('rho_prior', torch.as_tensor(rho_prior))

        # initialize mu and rho parameters for hidden layer's weights, and theta = logit(gamma)
        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim))   
        init.constant_(self.w_rho, -5)
        
        # initialize mu and rho parameters for hidden layer's biases
        self.v_mu = nn.Parameter(torch.Tensor(output_dim))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.v_mu, -bound, bound)
        self.v_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.v_rho, -5)

        self.tau_a_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.tau_b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.tau_a_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.tau_a_rho, -5)
        self.tau_b_rho = nn.Parameter(torch.Tensor(output_dim))
        init.constant_(self.tau_b_rho, -5)

        # initialize weight samples and binary indicators z
        self.w = None
        self.v = None

        # initialize kl for the hidden layer
        self.kl = 0

    def forward(self, X):
        """
            For one Monte Carlo (MC) sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        # sample weights and biases
        sigma_w = torch.log1p(torch.exp(self.w_rho))
        sigma_v = torch.log1p(torch.exp(self.v_rho))
        sigma_a_tau = torch.log1p(torch.exp(self.tau_a_rho))
        sigma_b_tau = torch.log1p(torch.exp(self.tau_b_rho))
        sigma_prior = torch.log1p(torch.exp(self.rho_prior)) 
        sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
        sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho))
        
        epsilon_w = torch.zeros_like(self.w_mu).normal_()
        epsilon_v = torch.zeros_like(self.v_mu).normal_()
        
        self.w = self.w_mu + sigma_w * epsilon_w
        self.v = self.v_mu + sigma_v * epsilon_v
        output = torch.mm(X, self.w) + self.v.expand(X.size()[0], self.output_dim)

        # record KL at sampled weight and bias with sampled inclusion probabilities
        w_tau_a_mu = self.tau_a_mu.expand(self.input_dim, self.output_dim)
        v_tau_a_mu = self.tau_a_mu
        w_tau_b_mu = self.tau_b_mu.expand(self.input_dim, self.output_dim)
        v_tau_b_mu = self.tau_b_mu
        sigma_w_a_tau = sigma_a_tau.expand(self.input_dim, self.output_dim)
        sigma_v_a_tau = sigma_a_tau
        sigma_w_b_tau = sigma_b_tau.expand(self.input_dim, self.output_dim)
        sigma_v_b_tau = sigma_b_tau
        
        kl_w = torch.log(sigma_prior) - torch.log(sigma_w) - 0.5 + \
                    0.5*(w_tau_a_mu + w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                    0.5*((sigma_w**2+self.w_mu**2)/sigma_prior**2)* \
                    torch.exp(-(w_tau_a_mu + w_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                        0.5*(sigma_w_a_tau**2 + sigma_w_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2))
        
        kl_v = torch.log(sigma_prior) - torch.log(sigma_v) - 0.5 + \
                        0.5*(v_tau_a_mu + v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                        0.5*((sigma_v**2+self.v_mu**2)/sigma_prior**2)* \
                        torch.exp(-(v_tau_a_mu + v_tau_b_mu + self.sig_a_mu + self.sig_b_mu) + \
                            0.5*(sigma_v_a_tau**2 + sigma_v_b_tau**2 + sig_a_sigma**2 + sig_b_sigma**2))
        
        kl_a_tau = torch.exp(self.tau_a_mu + 0.5*sigma_a_tau**2) - \
                        0.5*(self.tau_a_mu + 2*torch.log(sigma_a_tau) + 1.69315)

        kl_b_tau = torch.exp(0.5*sigma_b_tau**2 - self.tau_b_mu) - \
                        0.5*(2*torch.log(sigma_b_tau) - self.tau_b_mu + 1.69315)

        self.kl = torch.sum(kl_w) + torch.sum(kl_v) + torch.sum(kl_a_tau) + torch.sum(kl_b_tau)       

        return output
