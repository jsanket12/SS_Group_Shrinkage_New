import argparse

import torch
import numpy as np
import time
import torch.nn as nn
import pandas as pd 
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter # TensorBoard support
import transforms
import torchvision.datasets as datasets
from IPython import display
# import modules to build RunBuilder and RunManager helper classes
from collections  import OrderedDict
import os
import errno
import sys
sys.path.insert(1, '/lcrc/project/FastBayes/sanket_bnn/Resnet_expts/')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from GHS_resnet_models_non_center_reg_new import SS_GHS_Node_ResNet
from GHS_linear_layers_non_center_reg import GHS_layer
from GHS_Conv_layers_non_center_reg_new import SS_GHS_Node_Conv2d_layer
from GHS_BN_layers_non_center_reg import GHS_VB_BatchNorm2d

parser = argparse.ArgumentParser(description='Cifar10 ResNet')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--results_path', default='./results_resnet_new/cifar10/SS_GHS_new/', type = str, help = 'base path for saving result')
parser.add_argument('--warm_start_path', default='./results_resnet/cifar10/Freq/', type = str, help = 'base path for saving result')

# Resnet Architecture
parser.add_argument('--depth', default=20, type=int, help='depth of the resnet')

# Data setting
parser.add_argument('--num_classes', default = 10, type = int, help = 'total number of classes in classification dataset')

# Random Erasing
parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')

# Training Setting
parser.add_argument('--nepoch', default = 300, type = int, help = 'total number of training epochs')
parser.add_argument('--lr_decay_time', default = [150, 225], type = int, nargs= '+', help = 'when to multiply lr by 0.1')
parser.add_argument('--lr_decay_factor', default = 10, type = float, help = 'what factor to divide lr with')
parser.add_argument('--init_lr', default = 0.1, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
parser.add_argument('--batch_train', default = 128, type = int, help = 'batch size for training')
parser.add_argument('--num_MC_train', default = 1, type = int, help = 'Number of MC samples for training')
parser.add_argument('--num_MC_test', default = 1, type = int, help = 'Number of MC samples for testing')
parser.add_argument('--beta_anneal', default = 0, type=int, help='steps for annealing beta from 0 to 1')

# Optimizer Setting
parser.add_argument('--clip', default = 1, type = float, help = 'Gradient clipping value')

# Prior setting
parser.add_argument('--sigma_0', default = 1, type = float, help = 'sigma_0^2 in prior')
parser.add_argument('--c_a', default = 2, type = float, help = 'c_a in prior')
parser.add_argument('--c_b', default = 6, type = float, help = 'c_b in prior')
parser.add_argument('--c_reg', default = 1, type = float, help = 'c value')
parser.add_argument('--tau_0', default = 1, type = float, help = 'scale of global half-Cauchy prior')
parser.add_argument('--tau_1', default = 1, type = float, help = 'scale of local half-Cauchy prior')
parser.add_argument('--temp', default = 0.5, type = float, help = 'temperature')
parser.add_argument('--gamma_prior', default = 0.0001, type = float, help = 'prior inclusion probaility for filters/nodes')

# Fine tuning
parser.add_argument('--fine_tune_nepoch', default = 100, type = int, help = 'total number of fine tuning epochs')
parser.add_argument('--fine_tune_lr', default = 0.001, type = float, help = 'fine tune learning rate')

args = parser.parse_args()

writer = SummaryWriter()

class RunManager():
    def __init__(self):
        # tracking every epoch count, loss, accuracy, time
        self.epoch_start_time = None
        self.run_data = []

    def begin_run(self):
        self.run_start_time = time.time()

    def begin_epoch(self):
        self.epoch_start_time = time.time()

    def end_epoch(self,epoch,train_loss,train_accuracy,test_loss,test_accuracy,edge_sparsity,param_pruned,flops_ratio,flops_pruned,learning_rate,batch_size):
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
        results["param pruned"] = param_pruned
        results["flops ratio"] = flops_ratio
        results["flops pruned"] = flops_pruned
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

def linear_anneal(x, start, end, steps):
    assert x >= 0
    assert steps > 0
    assert start >= 0
    assert end >= 0
    if x > steps:
        return end
    if x < 0:
        return start
    return start + (end - start) / steps * x

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_dim = args.num_classes
    # num_MC_train = args.num_MC_train
    num_MC_test = args.num_MC_test

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize,
                                          transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1)])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         normalize])

    train_set = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_train, shuffle=True,num_workers=8,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=8,pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    m = RunManager()
    m.begin_run()

    sigma_0 = torch.as_tensor(args.sigma_0).to(device)
    c_a = torch.as_tensor(args.c_a).to(device)
    c_b = torch.as_tensor(args.c_b).to(device)
    c_reg = torch.as_tensor(args.c_reg).to(device)
    tau_0 = torch.as_tensor(args.tau_0).to(device)
    tau_1 = torch.as_tensor(args.tau_1).to(device)
    temp = torch.as_tensor(args.temp).to(device)
    gamma_prior = torch.tensor(args.gamma_prior).to(device)

    loss_func = nn.CrossEntropyLoss().to(device)

    net = SS_GHS_Node_ResNet(depth=args.depth, num_classes=target_dim, temp = temp, gamma_prior = gamma_prior, sigma_0 = sigma_0, 
                            tau_0 = tau_0, tau_1 = tau_1, c_a = c_a, c_b = c_b, c_reg = c_reg).to(device)
    # net.load_state_dict(torch.load(args.warm_start_path + 'best_resnet'+ str(args.depth) + '_freq_model_128.pt'), strict=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=0)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr)
    learning_rate = args.init_lr

    # total_num_para = 0
    # for module in net.modules():
    #     if isinstance(module, (SS_GHS_Node_Conv2d_layer,GHS_layer)):
    #         total_num_para += module.w_mu.numel()
    #         if module.v_mu is not None:
    #             total_num_para += module.v_mu.numel()

    PATH = args.results_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    num_epochs = args.nepoch
    train_Loss = []
    train_Accuracy = []
    test_Loss = []
    test_Accuracy = []
    Edge_sparsity = []
    flops_ratio = []
    flops_pruned = []
    param_pruned = []

    if args.depth == 20:
        total_num_para = 268346
        total_flops = 40551050
    elif args.depth == 32:
        total_num_para = 461882
        total_flops = 68862602
    elif args.depth == 44:
        total_num_para = 655418
        total_flops = 97174154
    elif args.depth == 56:
        total_num_para = 848954
        total_flops = 125485706
    elif args.depth == 110:
        total_num_para = 1719866
        total_flops = 252887690

    beta = 1.

    best_accuracy = 0
    NTrain = len(train_loader.dataset)

    for epoch in range(num_epochs):
        print('----------Epoch {}----------------'.format(epoch))
        m.begin_epoch()
        net.train()
        train_loss = 0.
        correct_train = 0

        if epoch in args.lr_decay_time:
            for para in optimizer.param_groups:
                para['lr'] = para['lr'] / args.lr_decay_factor
                learning_rate = para['lr']

        # Loss with beta
        if args.beta_anneal != 0:
            beta = linear_anneal(epoch, 0.0, 1.0, args.beta_anneal)

        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)

            # nll_train = 0.
            # outputs = torch.zeros(num_MC_train, labels.shape[0], target_dim).to(device)
            # for it in range(num_MC_train):
            #     outputs[it] = net(images) 
            #     nll_train += loss_func(outputs[it], labels)
            # output_mean = outputs.mean(dim=0) 
            output = net(images)
            nll_train = loss_func(output, labels)
            kl_train = 0.
            for module in net.modules():
                if isinstance(module, (SS_GHS_Node_Conv2d_layer,GHS_VB_BatchNorm2d,GHS_layer)):
                    kl_train += module.kl.div(NTrain)
            kl_train += net.kl_sig_and_c.div(NTrain)
            # loss = (nll_train/float(num_MC_train)) + beta*kl_train
            loss = nll_train + beta*kl_train

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            # train_loss += nll_train.div(num_MC_train).mul(images.shape[0]).item()
            # correct_train += output_mean.data.argmax(1).eq(labels.data).sum().item()
            train_loss += nll_train.mul(images.shape[0]).item()
            correct_train += output.data.argmax(1).eq(labels.data).sum().item()

        with torch.no_grad():
            net.eval()
            train_accuracy = correct_train / len(train_set)
            train_loss = train_loss/ len(train_set)

            train_Loss.append(train_loss)
            train_Accuracy.append(train_accuracy)         

            test_loss = 0
            prev = 0
            param_no_list = []
            flop_list = []
            outputs = torch.zeros(num_MC_test, len(test_set), target_dim).to(device)
            final_labels = torch.empty(len(test_set)).to(device)   
            for cnt, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device) 
                final_labels[prev:prev+labels.shape[0]] = labels    
                nll_test_comp = 0        
                for it in range(num_MC_test):       
                    outputs[it,prev:prev+labels.shape[0],:] = net(images)                    
                    nll_test_comp += loss_func(outputs[it,prev:prev+labels.shape[0],:], labels)      
                    param_overall = 0.
                    flops_overall = 0.                    
                    for name, module in net.named_modules():
                        if 'downsample' not in name:
                            if isinstance(module, (SS_GHS_Node_Conv2d_layer)): 
                                in_prune_channels = module.in_channels                   
                                out_prune_channels = (module.z != 0).sum()
                                if module.v is not None:
                                    param_overall += (module.w != 0).sum() + (module.v != 0).sum()
                                    flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1]+1)* \
                                                    (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups
                                else:
                                    param_overall += (module.w != 0).sum()
                                    flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1])* \
                                                    (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups                        
                            elif isinstance(module, (GHS_layer)):
                                if module.v is not None:
                                    param_overall += (module.input_dim+1)*module.output_dim
                                    flops_overall += (module.input_dim+1)*module.output_dim
                                else:
                                    param_overall += module.input_dim*module.output_dim
                                    flops_overall += module.input_dim*module.output_dim
                    param_no_list.append(param_overall.item())
                    flop_list.append(flops_overall.item())
                test_loss += nll_test_comp.div(num_MC_test).mul(images.shape[0]).item()
                prev += labels.shape[0] 
            test_loss = test_loss / len(test_set)
            output_mean = outputs.mean(dim=0)
            test_accuracy = output_mean.data.argmax(1).eq(final_labels.data).sum().div(len(test_set)).item()

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), PATH + 'SS_GHS_Node_non_center_reg_fine_tune_RN'+str(args.depth)+'_SGD_batch_'+str(args.batch_train)+'_sig0_'+str(args.sigma_0)+'_seed_'+str(args.seed) + '_best_model.pt')

            print('best accuracy:', best_accuracy)
            
            flops_pruned_val = np.median(flop_list)
            param_pruned_val = np.median(param_no_list)
            sparsity_overall = param_pruned_val/total_num_para
            flops_ratio_val = flops_pruned_val/total_flops            

            test_Loss.append(test_loss)
            test_Accuracy.append(test_accuracy)
            flops_pruned.append(flops_pruned_val)
            param_pruned.append(param_pruned_val)
            flops_ratio.append(flops_ratio_val)
            Edge_sparsity.append(sparsity_overall)

        writer.add_scalar('data/loss_train', train_loss, epoch)
        writer.add_scalar('data/accuracy_train', train_accuracy, epoch)
        writer.add_scalar('data/loss_test', test_loss, epoch)
        writer.add_scalar('data/accuracy_test', test_accuracy, epoch)
        writer.add_scalar('data/sparsity_edge', sparsity_overall, epoch)
        writer.add_scalar('data/param_pruned', param_pruned_val, epoch)
        writer.add_scalar('data/flops_ratio', flops_ratio_val, epoch)
        writer.add_scalar('data/flops_pruned', flops_pruned_val, epoch)
        writer.add_scalar('data/learning_rate', learning_rate, epoch)

        print('Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Edge sparsity: {}, FLOPs ratio: {}, Param pruned: {}, FLOPs pruned: {}'.format(
                            epoch, train_loss, train_accuracy, test_loss, test_accuracy, 
                            sparsity_overall,flops_ratio_val,param_pruned_val,flops_pruned_val))

        m.end_epoch(epoch, train_loss, train_accuracy, test_loss, test_accuracy, 
                    sparsity_overall, param_pruned_val, flops_ratio_val, flops_pruned_val, learning_rate, args.batch_train)

    print('------------------Finished sparse training--------------------------')
    print('-------------------------Fine tuning--------------------------------')

    net.load_state_dict(torch.load(PATH + 'SS_GHS_Node_non_center_reg_fine_tune_RN'+str(args.depth)+'_SGD_batch_'+str(args.batch_train)+'_sig0_'+str(args.sigma_0)+'_seed_'+str(args.seed) + '_best_model.pt'))

    optimizer = torch.optim.SGD(net.parameters(), lr=args.fine_tune_lr, momentum=args.momentum, weight_decay=0)

    net.fine_tune_flag()
    print(net.fine_tune)

    for name, para in net.named_parameters():
        if 'theta' in name:
            para.requires_grad = False

    for epoch_fine in range(args.fine_tune_nepoch):
        epoch = epoch_fine + num_epochs
        print('----------Fine Tune Epoch {}----------------'.format(epoch))
        m.begin_epoch()
        net.train()
        train_loss = 0.
        correct_train = 0    

        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)

            output = net(images)
            nll_train = loss_func(output, labels)
            kl_train = 0.
            for module in net.modules():
                if isinstance(module, (SS_GHS_Node_Conv2d_layer,GHS_VB_BatchNorm2d,GHS_layer)):
                    kl_train += module.kl.div(NTrain)
            kl_train += net.kl_sig_and_c.div(NTrain)
            loss = nll_train + kl_train

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            train_loss += nll_train.mul(images.shape[0]).item()
            correct_train += output.data.argmax(1).eq(labels.data).sum().item()

        with torch.no_grad():
            net.eval()
            train_accuracy = correct_train / len(train_set)
            train_loss = train_loss/ len(train_set)

            train_Loss.append(train_loss)
            train_Accuracy.append(train_accuracy)         

            test_loss = 0
            prev = 0
            param_no_list = []
            flop_list = []
            outputs = torch.zeros(num_MC_test, len(test_set), target_dim).to(device)
            final_labels = torch.empty(len(test_set)).to(device)   
            for cnt, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device) 
                final_labels[prev:prev+labels.shape[0]] = labels    
                nll_test_comp = 0        
                for it in range(num_MC_test):       
                    outputs[it,prev:prev+labels.shape[0],:] = net(images)                    
                    nll_test_comp += loss_func(outputs[it,prev:prev+labels.shape[0],:], labels)      
                    param_overall = 0.
                    flops_overall = 0.                    
                    for name, module in net.named_modules():
                        if 'downsample' not in name:
                            if isinstance(module, (SS_GHS_Node_Conv2d_layer)): 
                                in_prune_channels = module.in_channels                   
                                out_prune_channels = (module.z != 0).sum()
                                # print(name, (module.z == 0).sum())
                                if module.v is not None:
                                    param_overall += (module.w != 0).sum() + (module.v != 0).sum()
                                    flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1]+1)* \
                                                    (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups
                                else:
                                    param_overall += (module.w != 0).sum()
                                    flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1])* \
                                                    (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups                        
                            elif isinstance(module, (GHS_layer)):
                                if module.v is not None:
                                    param_overall += (module.input_dim+1)*module.output_dim
                                    flops_overall += (module.input_dim+1)*module.output_dim
                                else:
                                    param_overall += module.input_dim*module.output_dim
                                    flops_overall += module.input_dim*module.output_dim
                    param_no_list.append(param_overall.item())
                    flop_list.append(flops_overall.item())
                test_loss += nll_test_comp.div(num_MC_test).mul(images.shape[0]).item()
                prev += labels.shape[0] 
            test_loss = test_loss / len(test_set)
            output_mean = outputs.mean(dim=0)
            test_accuracy = output_mean.data.argmax(1).eq(final_labels.data).sum().div(len(test_set)).item()

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), PATH + 'SS_GHS_Node_non_center_reg_fine_tune_RN'+str(args.depth)+'_SGD_batch_'+str(args.batch_train)+'_sig0_'+str(args.sigma_0)+'_seed_'+str(args.seed) + '_final_model.pt')

            print('best accuracy:', best_accuracy)
            
            flops_pruned_val = np.median(flop_list)
            param_pruned_val = np.median(param_no_list)
            sparsity_overall = param_pruned_val/total_num_para
            flops_ratio_val = flops_pruned_val/total_flops            

            test_Loss.append(test_loss)
            test_Accuracy.append(test_accuracy)
            flops_pruned.append(flops_pruned_val)
            param_pruned.append(param_pruned_val)
            flops_ratio.append(flops_ratio_val)
            Edge_sparsity.append(sparsity_overall)

        writer.add_scalar('data/loss_train', train_loss, epoch)
        writer.add_scalar('data/accuracy_train', train_accuracy, epoch)
        writer.add_scalar('data/loss_test', test_loss, epoch)
        writer.add_scalar('data/accuracy_test', test_accuracy, epoch)
        writer.add_scalar('data/sparsity_edge', sparsity_overall, epoch)
        writer.add_scalar('data/param_pruned', param_pruned_val, epoch)
        writer.add_scalar('data/flops_ratio', flops_ratio_val, epoch)
        writer.add_scalar('data/flops_pruned', flops_pruned_val, epoch)
        writer.add_scalar('data/learning_rate', learning_rate, epoch)

        print('Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Edge sparsity: {}, FLOPs ratio: {}, Param pruned: {}, FLOPs pruned: {}'.format(
                            epoch, train_loss, train_accuracy, test_loss, test_accuracy, 
                            sparsity_overall,flops_ratio_val,param_pruned_val,flops_pruned_val))

        m.end_epoch(epoch, train_loss, train_accuracy, test_loss, test_accuracy, 
                    sparsity_overall, param_pruned_val, flops_ratio_val, flops_pruned_val, learning_rate, args.batch_train)

    print('Finished Training')
    writer.close()
    
    m.save(PATH,'results_SS_GHS_Node_non_center_reg_fine_tune_RN'+str(args.depth)+'_SGD_batch_'+str(args.batch_train)+'_sig0_'+str(args.sigma_0)+'_seed_'+str(args.seed)) 

    # torch.save(net.state_dict(), PATH + 'SS_GHS_Node_non_center_reg_RN'+str(args.depth)+'_SGD_batch_'+str(args.batch_train)+'_sig0_'+str(args.sigma_0) + '_model.pt')

    num_epochs = args.nepoch + args.fine_tune_nepoch
    
    plt.plot(range(num_epochs), train_Loss, 'b', label='Train')                
    plt.plot(range(num_epochs), test_Loss, 'orange', label='Test')
    plt.title('SS_GHS_Node_non_center_reg: Train-Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.ylim([0.55, 1.0])
    plt.legend(loc='upper right', frameon=False)
    plt.grid(ls='dotted')
    plt.savefig(os.path.join(PATH,'SS_GHS_Node_non_center_reg_fine_tune_RN'+str(args.depth)+'_SGD_batch_'+str(args.batch_train)+'_sig0_'+str(args.sigma_0)+'_loss.png'),dpi=300)
    plt.close()

    plt.plot(range(num_epochs), train_Accuracy, 'b', label='Train')
    plt.plot(range(num_epochs), test_Accuracy, 'orange', label='Test')
    plt.title('SS_GHS_Node_non_center_reg: Train-Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.ylim([0.55, 1.0])
    plt.legend(loc='upper right', frameon=False)
    plt.grid(ls='dotted')
    plt.savefig(os.path.join(PATH,'SS_GHS_Node_non_center_reg_fine_tune_RN'+str(args.depth)+'_SGD_batch_'+str(args.batch_train)+'_sig0_'+str(args.sigma_0)+'_accuracy.png'),dpi=300)
    plt.close()

    plt.plot(range(num_epochs), Edge_sparsity, 'g')
    plt.title('SS_GHS_Node_non_center_reg: Edge Sparsity')
    plt.xlabel('Epochs')
    plt.ylabel('Sparsity')
    # plt.ylim([0.65, 1.0])
    plt.legend(loc='upper right', frameon=False)
    plt.grid(ls='dotted')
    plt.savefig(os.path.join(PATH,'SS_GHS_Node_non_center_reg_fine_tune_RN'+str(args.depth)+'_SGD_batch_'+str(args.batch_train)+'_sig0_'+str(args.sigma_0)+'_edge_sparsity.png'),dpi=300)
    plt.close()

if __name__ == '__main__':
    main()