import argparse

import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter # TensorBoard support
# import torchvision.transforms as transforms
import transforms
import torchvision.datasets as datasets
from IPython import display
# import modules to build RunBuilder and RunManager helper classes
from collections  import OrderedDict
# from itertools import product
import os
import errno
import sys
sys.path.insert(1, '/mnt/home/jantresa/SS-IG_New/')
from GHS_Conv_layers_non_center import SS_GHS_Node_Conv2d_layer, GHS_Conv2d_layer
from GHS_linear_layers_non_center import SS_GHS_Node_layer, GHS_layer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# PATH = './results_lenet/mnist/SS_GHS/'

parser = argparse.ArgumentParser(description='MNIST Lenet')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
# parser.add_argument('--results_path', default='./results_resnet/cifar10/SS_Gauss_Node/', type = str, help = 'base path for saving result')

# Data setting
parser.add_argument('--num_classes', default = 10, type = int, help = 'total number of classes in classification dataset')

# Training Setting
parser.add_argument('--nepoch', default = 300, type = int, help = 'total number of training epochs')
parser.add_argument('--init_lr', default = 0.1, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
parser.add_argument('--batch_train', default = 128, type = int, help = 'batch size for training')
parser.add_argument('--num_MC_train', default = 1, type = int, help = 'Number of MC samples for training')
parser.add_argument('--num_MC_test', default = 5, type = int, help = 'Number of MC samples for testing')

# Optimizer Setting
parser.add_argument('--clip', default = 1, type = float, help = 'Gradient clipping value')

# Prior Setting
parser.add_argument('--sigma_0', default = 1, type = float, help = 'sigma_0^2 in prior')
parser.add_argument('--temp', default = 0.5, type = float, help = 'temperature')
parser.add_argument('--tau_0', default = 1e-5, type = float, help = 'scale of global half-Cauchy prior')

args = parser.parse_args()

writer = SummaryWriter()

#### sparsefunc file content
class SFunc(nn.Module):
    """
        Our BNN
    """
    def __init__(self, hidden_dim1, hidden_dim2, target_dim, temp, gamma_prior, gamma_prior_star1, gamma_prior_star2, sigma_0, tau_0):

        # initialize the network using the MLP layer
        super().__init__()
        self.tau_0 = tau_0
        self.rho_prior = torch.as_tensor([np.log(np.exp(1.3) - 1)])
        # self.register_buffer('tau_0', torch.as_tensor(tau_0))
        self.register_buffer('temp', torch.as_tensor(temp))
        self.register_buffer('gamma_prior', torch.as_tensor(gamma_prior))
        self.register_buffer('gamma_prior_star1', torch.as_tensor(gamma_prior_star1))
        self.register_buffer('gamma_prior_star2', torch.as_tensor(gamma_prior_star2))

        self.sig_a_mu = nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        self.sig_a_rho = nn.Parameter(torch.Tensor(1).uniform_(-6., -6.))
        self.sig_b_mu = nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        self.sig_b_rho = nn.Parameter(torch.Tensor(1).uniform_(-6., -6.))  

        # Spike-and-slab node selection with horseshoe
        # self.conv1 = SS_GHS_Node_Conv2d_layer(in_channels=1, out_channels=20, rho_prior=self.rho_prior, temp=self.temp, gamma_prior=self.gamma_prior_star1, 
        #                                         sig_a_mu = self.sig_a_mu, sig_a_rho = self.sig_a_rho, sig_b_mu = self.sig_b_mu, sig_b_rho = self.sig_b_rho, 
        #                                          kernel_size=5, stride=1)
        # self.conv2 = SS_GHS_Node_Conv2d_layer(in_channels=20, out_channels=50, rho_prior=self.rho_prior, temp=self.temp, gamma_prior=self.gamma_prior_star2,
        #                                         sig_a_mu = self.sig_a_mu, sig_a_rho = self.sig_a_rho, sig_b_mu = self.sig_b_mu, sig_b_rho = self.sig_b_rho, 
        #                                          kernel_size=5, stride=1)
        # self.l1 = SS_GHS_Node_layer(4*4*50, hidden_dim1, self.rho_prior, self.temp, self.gamma_prior_star1, 
        #                                     self.sig_a_mu, self.sig_a_rho, self.sig_b_mu, self.sig_b_rho)
        # self.l2 = SS_GHS_Node_layer(hidden_dim1, hidden_dim2, self.rho_prior, self.temp, self.gamma_prior_star2, 
        #                                     self.sig_a_mu, self.sig_a_rho, self.sig_b_mu, self.sig_b_rho)
        # self.l4 = GHS_layer(hidden_dim2, target_dim, self.rho_prior, self.sig_a_mu, self.sig_a_rho, 
        #                                     self.sig_b_mu, self.sig_b_rho)

        # Horseshoe without spike-and-slab
        # self.conv1 = GHS_Conv2d_layer(in_channels=1, out_channels=20, rho_prior=self.rho_prior, sig_a_mu = self.sig_a_mu, 
        #                                         sig_a_rho = self.sig_a_rho, sig_b_mu = self.sig_b_mu, sig_b_rho = self.sig_b_rho, 
        #                                         kernel_size=5, stride=1)
        # self.conv2 = GHS_Conv2d_layer(in_channels=20, out_channels=50, rho_prior=self.rho_prior, sig_a_mu = self.sig_a_mu, 
        #                                         sig_a_rho = self.sig_a_rho, sig_b_mu = self.sig_b_mu, sig_b_rho = self.sig_b_rho, 
        #                                         kernel_size=5, stride=1)
        # self.l1 = GHS_layer(4*4*50, hidden_dim1, self.rho_prior, self.sig_a_mu, self.sig_a_rho, 
        #                                     self.sig_b_mu, self.sig_b_rho)
        # self.l2 = GHS_layer(hidden_dim1, hidden_dim2, self.rho_prior, self.sig_a_mu, self.sig_a_rho, 
        #                                     self.sig_b_mu, self.sig_b_rho)
        # self.l4 = GHS_layer(hidden_dim2, target_dim, self.rho_prior, self.sig_a_mu, self.sig_a_rho, 
        #                                     self.sig_b_mu, self.sig_b_rho)  

        # Spike-and-slab node selection with horseshoe
        self.conv1 = SS_GHS_Node_Conv2d_layer(1, 20, kernel_size=5, stride=1, temp=temp, gamma_prior=gamma_prior_star1, sigma_0=sigma_0)
        self.conv2 = SS_GHS_Node_Conv2d_layer(20, 50, kernel_size=5, stride=1, temp=temp, gamma_prior=gamma_prior_star2, sigma_0=sigma_0)
        self.l1 = SS_GHS_Node_layer(4*4*50, hidden_dim1, temp=temp, gamma_prior=gamma_prior_star1, sigma_0=sigma_0)
        self.l2 = SS_GHS_Node_layer(hidden_dim1, hidden_dim2, temp=temp, gamma_prior=gamma_prior_star2, sigma_0=sigma_0)
        self.l4 = GHS_layer(hidden_dim2, target_dim, sigma_0=sigma_0)

        # Horseshoe without spike-and-slab
        # self.conv1 = GHS_Conv2d_layer(1, 20, sig_a_mu = self.sig_a_mu, sig_a_rho = self.sig_a_rho, 
        #                                         sig_b_mu = self.sig_b_mu, sig_b_rho = self.sig_b_rho,
        #                                         kernel_size=5, stride=1, sigma_0=sigma_0)
        # self.conv2 = GHS_Conv2d_layer(20, 50, sig_a_mu = self.sig_a_mu, sig_a_rho = self.sig_a_rho, 
        #                                         sig_b_mu = self.sig_b_mu, sig_b_rho = self.sig_b_rho,
        #                                         kernel_size=5, stride=1, sigma_0=sigma_0)
        # self.l1 = GHS_layer(4*4*50, hidden_dim1, sig_a_mu = self.sig_a_mu, sig_a_rho = self.sig_a_rho, 
        #                                     sig_b_mu = self.sig_b_mu, sig_b_rho = self.sig_b_rho, sigma_0=sigma_0)
        # self.l2 = GHS_layer(hidden_dim1, hidden_dim2, sig_a_mu = self.sig_a_mu, sig_a_rho = self.sig_a_rho, 
        #                                     sig_b_mu = self.sig_b_mu, sig_b_rho = self.sig_b_rho, sigma_0=sigma_0)
        # self.l4 = GHS_layer(hidden_dim2, target_dim, sig_a_mu = self.sig_a_mu, sig_a_rho = self.sig_a_rho, 
        #                                     sig_b_mu = self.sig_b_mu, sig_b_rho = self.sig_b_rho, sigma_0=sigma_0)

        self.kl_sig = 0.

    def forward(self, X):
        """
            output of the BNN for one Monte Carlo sample
            :param X: [batch_size, data_dim]
            :return: [batch_size, target_dim]
        """
        sig_a_sigma = torch.log1p(torch.exp(self.sig_a_rho))
        sig_b_sigma = torch.log1p(torch.exp(self.sig_b_rho))

        sigma_sig = 0.5*torch.sqrt(sig_a_sigma**2 + sig_b_sigma**2)
        epsilon_sig = torch.zeros_like(self.sig_a_mu).normal_()
        Global_scale = torch.sqrt(torch.exp(0.5*(self.sig_a_mu+self.sig_b_mu) + sigma_sig * epsilon_sig))

        # kl_a_sig = -torch.log(self.tau_0) + torch.exp(self.sig_a_mu + 0.5*sig_a_sigma**2)/self.tau_0 - \
        #                 0.5*(self.sig_a_mu + 2*torch.log(sig_a_sigma) + 1.69315)
        kl_a_sig =  torch.exp(self.sig_a_mu + 0.5*sig_a_sigma**2) - \
                        0.5*(self.sig_a_mu + 2*torch.log(sig_a_sigma) + 1.69315)
        
        kl_b_sig = torch.exp(0.5*sig_b_sigma**2 - self.sig_b_mu) - \
                        0.5*(2*torch.log(sig_b_sigma) - self.sig_b_mu + 1.69315)

        self.kl_sig = torch.sum(kl_a_sig) + torch.sum(kl_b_sig)

        # print("\tIn Model: input size", X.size())
        output = F.silu(F.max_pool2d(self.conv1({0:X, 1:Global_scale})[0], 2))
        output = F.silu(F.max_pool2d(self.conv2({0:output, 1:Global_scale})[0], 2))
        output = F.silu(self.l1({0:output.reshape(-1,4*4*50), 1:Global_scale})[0])
        output = F.silu(self.l2({0:output, 1:Global_scale})[0])
        output = self.l4({0:output, 1:Global_scale})[0]
        
        return output

class RunManager():
    def __init__(self):
        # tracking every epoch count, loss, accuracy, time
        self.epoch_start_time = None
        self.run_data = []

    def begin_run(self):
        self.run_start_time = time.time()

    def begin_epoch(self):
        self.epoch_start_time = time.time()

    def end_epoch(self,epoch,train_loss,train_accuracy,test_loss,test_accuracy,edge_sparsity,learning_rate,batch_size):
        # calculate epoch duration and run duration(accumulate)
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        # Write into 'results' (OrderedDict) for all run related data
        results = OrderedDict()
        results["epoch"] = epoch
        results["Train loss"] = train_loss    #loss
        results["Test loss"] = test_loss
        results["Train Accuracy"] = train_accuracy   #accuracy
        results["Test Accuracy"] = test_accuracy
        results["edge sparsity"] = edge_sparsity
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        results["lr"] = learning_rate
        results["batch_size"] = batch_size

        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')
        # display epoch information and show progress
        display.clear_output(wait=True)
        display.display(df)
  
    def save(self, Path, fileName):
        pd.DataFrame.from_dict(
            self.run_data, 
            orient = 'columns',
        ).to_csv(f'{Path+fileName}.csv')

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x * 255. / 126.)])

    train_set = datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_train, shuffle=True,num_workers=8,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=8,pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    m = RunManager()
    m.begin_run()

    data_size = len(train_set)
    data_dim = 4*4*50
    hidden_dim1 = 800
    hidden_dim2 = 500
    target_dim = args.num_classes

    L=2
    u_0= ((L+1)**2)*(np.log(data_size) + np.log(L+1) + np.log(data_dim+1) + np.log(hidden_dim1))
    u_1= ((L+1)**2)*(np.log(data_size) + np.log(L+1) + np.log(hidden_dim1+1) + np.log(hidden_dim2))
    u_2= ((L+1)**2)*(np.log(data_size) + np.log(L+1) + np.log(hidden_dim2+1) + np.log(target_dim))
    v_0=(data_dim+1)**2 + np.log(hidden_dim1+1) + np.log(hidden_dim2+1) + L + np.log(hidden_dim1) + \
            np.log(data_dim+1) + np.log(data_size) + np.log(u_0+u_1+u_2)
    v_1=(hidden_dim1+1)**2 + np.log(data_dim+1) + np.log(hidden_dim2+1) + L + np.log(hidden_dim2) + \
            np.log(hidden_dim1+1) + np.log(data_size) + np.log(u_0+u_1+u_2)
    a_hlayer1 = np.log(hidden_dim1) + 0.000000001*(data_dim+1)*v_0
    a_hlayer2 = np.log(hidden_dim2) + 0.000000001*(hidden_dim1+1)*v_1
    gamma_prior_star1 = torch.as_tensor(1/np.exp(a_hlayer1))
    gamma_prior_star2 = torch.as_tensor(1/np.exp(a_hlayer2)) 

    L=L+1
    total = (data_dim+1) * hidden_dim1  + (hidden_dim1+1)* hidden_dim2 + (hidden_dim2+1) * 10
    a = np.log(total) + 0.1*((L+1)*np.log(max(hidden_dim1,hidden_dim2)) + np.log(np.sqrt(data_size)*data_dim))
    lm = 1/np.exp(a)
    gamma_prior = torch.tensor(lm)

    print('lambda_Edge:',gamma_prior.item(),'lambda_Node_1:',gamma_prior_star1.item(),
            ', Lambda_Node_2:', gamma_prior_star2.item())

    gamma_prior = gamma_prior.to(device)
    gamma_prior_star1 = gamma_prior_star1.to(device)
    gamma_prior_star2 = gamma_prior_star2.to(device)

    tau_0 = torch.as_tensor(args.sigma_0).to(device)
    sigma_0 = args.sigma_0
    num_MC_train = args.num_MC_train
    num_MC_test = args.num_MC_test

    temp = torch.as_tensor(args.temp).to(device)

    loss_func = nn.CrossEntropyLoss().to(device)

    net = net = SFunc(hidden_dim1, hidden_dim2, target_dim, temp, gamma_prior, gamma_prior_star1, gamma_prior_star2, sigma_0, tau_0).to(device)

    # optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr)
    learning_rate = args.init_lr
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.init_lr/args.lr_decay_factor, max_lr=args.init_lr,cycle_momentum=False)

    total_num_para = 0
    for name, para in net.named_parameters():
        if 'w_mu' in name or 'v_mu' in name:
            total_num_para += para.numel() 
    print(total_num_para)

    # total_num_para = 0
    # for module in net.modules():
    #     if isinstance(module, (SSGauss_Node_Conv2d_layer,Gauss_layer)):
    #         total_num_para += module.w_mu.numel()
    #         if module.v_mu is not None:
    #             total_num_para += module.v_mu.numel()
    # print(total_num_para)

    # PATH = args.results_path
    # if not os.path.isdir(PATH):
    #     try:
    #         os.makedirs(PATH)
    #     except OSError as exc:  
    #         if exc.errno == errno.EEXIST and os.path.isdir(PATH):
    #             pass
    #         else:
    #             raise

    num_epochs = args.nepoch
    train_Loss = []
    train_Accuracy = []
    test_Loss = []
    test_Accuracy = []
    Edge_sparsity = []

    NTrain = len(train_loader.dataset)

    for epoch in range(num_epochs):
        print('----------Epoch {}----------------'.format(epoch))
        m.begin_epoch()
        net.train()
        train_loss = 0.
        correct_train = 0

        # if epoch in args.lr_decay_time:
        #     for para in optimizer.param_groups:
        #         para['lr'] = para['lr'] / args.lr_decay_factor
        #         learning_rate = para['lr']
        # learning_rate = adjust_learning_rate(optimizer,epoch)

        # for name, para in net.named_parameters():
        #         if 'theta' in name:
        #             if epoch < args.lr_decay_time[0]:
        #                 para.requires_grad = False
        #             else:
        #                 para.requires_grad = True

        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)

            nll_train = 0.
            outputs = torch.zeros(num_MC_train, labels.shape[0], target_dim).to(device)
            for it in range(num_MC_train):
                outputs[it] = net(images) 
                nll_train += loss_func(outputs[it], labels)
            output_mean = outputs.mean(dim=0)            
            kl_train = net.conv1.kl+net.conv2.kl+net.l1.kl+net.l2.kl+net.l4.kl+net.kl_sig
            loss = (nll_train/float(num_MC_train)) + kl_train.div(NTrain)          

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            # scheduler.step()

            train_loss += nll_train.div(num_MC_train).mul(images.shape[0]).item()
            correct_train += output_mean.data.argmax(1).eq(labels.data).sum().item()

        with torch.no_grad():
            net.eval()
            train_accuracy = correct_train / len(train_set)
            train_loss = train_loss/ len(train_set)

            train_Loss.append(train_loss)
            train_Accuracy.append(train_accuracy)

            nll_test_comp = 0
            test_loss = 0 
            outputs = torch.zeros(num_MC_test, len(test_set), target_dim).to(device)
            final_labels = torch.empty(len(test_set)).to(device)   
            for cnt, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device) 
                final_labels[cnt*labels.shape[0]:(cnt+1)*labels.shape[0]] = labels               
                for it in range(num_MC_test):       
                    outputs[it,cnt*labels.shape[0]:(cnt+1)*labels.shape[0],:] = net(images)                    
                    nll_test_comp += loss_func(outputs[it,cnt*labels.shape[0]:(cnt+1)*labels.shape[0],:], labels)        
                test_loss += nll_test_comp.div(num_MC_test).mul(images.shape[0]).item()
            test_loss = test_loss / len(test_set)
            output_mean = outputs.mean(dim=0)
            test_accuracy = output_mean.data.argmax(1).eq(labels.data).sum().div(len(test_set)).item()

            # if net.conv1.transposed:
            #     conv1_w = net.conv1.w.permute(1, 0, 2, 3)   
            #     conv1_w = conv1_w.view(-1, conv1_w.shape[1]*conv1_w.shape[2]*conv1_w.shape[3]).T
            # else:
            #     conv1_w = net.conv1.w 
            #     conv1_w = conv1_w.view(-1, conv1_w.shape[1]*conv1_w.shape[2]*conv1_w.shape[3]).T

            # if net.conv2.transposed:
            #     conv2_w = net.conv2.w.permute(1, 0, 2, 3)   
            #     conv2_w = conv2_w.view(-1, conv2_w.shape[1]*conv2_w.shape[2]*conv2_w.shape[3]).T
            # else:
            #     conv2_w = net.conv2.w 
            #     conv2_w = conv2_w.view(-1, conv2_w.shape[1]*conv2_w.shape[2]*conv2_w.shape[3]).T

            # arr1_l = torch.norm(torch.cat((conv1_w,net.conv1.v.expand(1, net.conv1.v.size()[0])),0),1,0)
            # one1_l = (arr1_l!=0).float()
            # sparsity_conv1_val = torch.sum(one1_l)/(arr1_l.size()[0])

            # arr2_l = torch.norm(torch.cat((conv2_w,net.conv2.v.expand(1, net.conv2.v.size()[0])),0),1,0)
            # one2_l = (arr2_l!=0).float()
            # sparsity_conv2_val = torch.sum(one2_l)/(arr2_l.size()[0])

            # arr1_l = torch.norm(torch.cat((net.l1.w,net.l1.v.expand(1, net.l1.v.size()[0])),0),1,0)
            # one1_l = (arr1_l!=0).float()
            # sparsity_mlp1_val = torch.sum(one1_l)/(arr1_l.size()[0])

            # arr2_l = torch.norm(torch.cat((net.l2.w,net.l2.v.expand(1, net.l2.v.size()[0])),0),1,0)
            # one2_l = (arr2_l!=0).float()
            # sparsity_mlp2_val = torch.sum(one2_l)/(arr2_l.size()[0])

            sparsity_overall = (torch.sum((net.conv1.w != 0).float()) + torch.sum((net.conv1.v != 0).float()) + 
                            torch.sum((net.conv2.w != 0).float()) + torch.sum((net.conv2.v != 0).float()) + 
                            torch.sum((net.l1.w != 0).float()) + torch.sum((net.l1.v != 0).float()) + 
                            torch.sum((net.l2.w != 0).float()) + torch.sum((net.l2.v != 0).float()) +
                            torch.sum((net.l4.w != 0).float()) + torch.sum((net.l4.v != 0).float())) / total_num_para
            sparsity_overall = sparsity_overall.item()

            # sparsity_overall = 0
            # for module in net.modules():
            #     if isinstance(module, (SSGauss_Node_Conv2d_layer,Gauss_layer)):
            #         sparsity_overall += (module.w != 0).sum()
            #         if module.v is not None:
            #             sparsity_overall += (module.v != 0).sum()
            # sparsity_overall = sparsity_overall.div(total_num_para).item()

            test_Loss.append(test_loss)
            test_Accuracy.append(test_accuracy)
            Edge_sparsity.append(sparsity_overall)

        writer.add_scalar('data/loss_train', train_loss, epoch)
        writer.add_scalar('data/accuracy_train', train_accuracy, epoch)
        writer.add_scalar('data/loss_test', test_loss, epoch)
        writer.add_scalar('data/accuracy_test', test_accuracy, epoch)
        writer.add_scalar('data/sparsity_edge', sparsity_overall, epoch)
        writer.add_scalar('data/learning_rate', learning_rate, epoch)

        print('Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Edge sparsity: {}'.format(
                            epoch, train_loss, train_accuracy, test_loss, test_accuracy, sparsity_overall))

        m.end_epoch(epoch, train_loss, train_accuracy,
                    test_loss, test_accuracy, sparsity_overall, learning_rate, args.batch_train)

    print('Finished Training')
    writer.close()
    
    # m.save(PATH,'results_SS_Gaussian_Node_Resnet_CIFAR10_silu_lr_decay_'+str(args.batch_train)) 

    # torch.save(net.state_dict(), PATH + 'SS_Gaussian_Node_Resnet_CIFAR10_model_silu_lr_decay_'+ str(args.batch_train) + '.pt')

    # plt.plot(range(num_epochs), train_Loss, 'b', label='Train')                
    # plt.plot(range(num_epochs), test_Loss, 'orange', label='Test')
    # plt.title('SS_Gaussian_Node: Train-Test Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # # plt.ylim([0.55, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,'SS_Gaussian_Node_Resnet_CIFAR10_loss_silu_lr_decay_'+str(args.batch_train)+'.png'),dpi=300)
    # plt.close()

    # plt.plot(range(num_epochs), train_Accuracy, 'b', label='Train')
    # plt.plot(range(num_epochs), test_Accuracy, 'orange', label='Test')
    # plt.title('SS_Gaussian_Node: Train-Test Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # # plt.ylim([0.55, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,'SS_Gaussian_Node_Resnet_CIFAR10_accuracy_silu_lr_decay_'+str(args.batch_train)+'.png'),dpi=300)
    # plt.close()

    # plt.plot(range(num_epochs), Edge_sparsity, 'g')
    # plt.title('SS_Gaussian_Node: Edge Sparsity')
    # plt.xlabel('Epochs')
    # plt.ylabel('Sparsity')
    # # plt.ylim([0.65, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,'SS_Gaussian_Node_Resnet_CIFAR10_sparsity_edge_silu_lr_decay_'+str(args.batch_train)+'.png'),dpi=300)
    # plt.close()


if __name__ == '__main__':
    main()