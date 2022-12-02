import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.utils.data

import pandas as pd 
import time
from IPython import display
# import modules to build RunBuilder and RunManager helper classes
from collections  import OrderedDict

#import torchvision.transforms as transforms
import transforms
import torchvision.datasets as datasets
import os
import errno
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(1, '/lcrc/project/FastBayes/sanket_bnn/Resnet_expts/')
import Sun2021_resnet_vb
from Sun2021_vb_net import VB_Linear, VB_Conv2d

parser = argparse.ArgumentParser(description='Cifar10 ResNet Compression')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--base_path', default='./Sun_et_al_2021/result/cifar/vb/', type = str, help = 'base path for saving result')
parser.add_argument('--model_path', default='test_run_VB/', type = str, help = 'folder name for saving model')
parser.add_argument('--fine_tune_path', default='fine_tune/', type = str, help = 'folder name for saving fine tune model')

# Resnet Architecture
parser.add_argument('--depth', default=20, type=int, help='Model depth.')

# Data setting
parser.add_argument('--num_classes', default = 10, type = int, help = 'total number of classes in classification dataset')

# Random Erasing
parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')

# Training Setting
parser.add_argument('--only_fine_tune', default = 0, type = int, help = 'only fine tune')
parser.add_argument('--nepoch', default = 300, type = int, help = 'total number of training epochs')
parser.add_argument('--lr_decay_time', default = [150, 225], type = int, nargs= '+', help = 'when to multiply lr by 0.1')
parser.add_argument('--lr_decay_factor', default = 10, type = float, help = 'what factor to divide lr with')
parser.add_argument('--init_lr', default = 0.1, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
parser.add_argument('--batch_train', default = 128, type = int, help = 'batch size for training')
# parser.add_argument('--batch_test', default = 128, type = int, help = 'batch size for testing')
parser.add_argument('--num_MC_test', default = 1, type = int, help = 'Number of MC samples for testing')


# Fine Tuning Setting
parser.add_argument('--nepoch_fine_tune', default = 100, type = int, help = 'total number of training epochs in fine tuning')
parser.add_argument('--lr_decay_time_fine_tune', default = [], type = int, nargs= '*', help = 'when to multiply lr by 0.1 in fine tuning')
parser.add_argument('--init_lr_fine_tune', default = 0.001, type = float, help = 'initial learning rate in fine tuning')
parser.add_argument('--momentum_fine_tune', default = 0.9, type = float, help = 'momentum in SGD in fine tuning')

# Prior Setting
parser.add_argument('--sigma0', default = 0.000004, type = float, help = 'sigma_0^2 in prior')
parser.add_argument('--sigma1', default = 0.04, type = float, help = 'sigma_1^2 in prior')
parser.add_argument('--lambdan', default = 0.0000001, type = float, help = 'lambda_n in prior')
parser.add_argument('--prune_ratio', default = 0.1, type = float, help = 'prune ratio in prior')

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

def model_eval(args, net, data_loader, device, loss_func):
    # net.eval()
    # correct = 0
    # total_loss = 0
    # total_count = 0
    # for cnt, (images, labels) in enumerate(data_loader):
    #     images, labels = images.to(device), labels.to(device)
    #     outputs = net(images)
    #     loss = loss_func(outputs, labels)
    #     prediction = outputs.data.max(1)[1]
    #     correct += prediction.eq(labels.data).sum().item()
    #     total_loss += loss.mul(images.shape[0]).item()
    #     total_count += images.shape[0]

    net.eval()
    NTest = len(data_loader.dataset)
    test_loss = 0
    prev = 0
    outputs = torch.zeros(args.num_MC_test, NTest, args.num_classes).to(device)
    final_labels = torch.empty(NTest).to(device)   
    for _ , (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device) 
        final_labels[prev:prev+labels.shape[0]] = labels    
        nll_test_comp = 0        
        for it in range(args.num_MC_test):       
            outputs[it,prev:prev+labels.shape[0],:] = net(images)                    
            nll_test_comp += loss_func(outputs[it,prev:prev+labels.shape[0],:], labels)      
        test_loss += nll_test_comp.div(args.num_MC_test).mul(images.shape[0]).item()
        prev += labels.shape[0] 
    test_loss = test_loss / NTest
    output_mean = outputs.mean(dim=0)
    test_accuracy = output_mean.data.argmax(1).eq(final_labels.data).sum().div(NTest).item()

    # return  1.0 * correct / total_count, total_loss / total_count
    return test_accuracy, test_loss

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize,
                                          transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3)])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         normalize])

    train_set = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=test_transform)

    np.random.seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_train, shuffle=True,num_workers=8,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False,num_workers=8,pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    m = RunManager()
    m.begin_run()

    loss_func = nn.CrossEntropyLoss().to(device)

    lambda_n = args.lambdan
    prior_sigma_0 = args.sigma0
    prior_sigma_1 = args.sigma1

    net = Sun2021_resnet_vb.ResNet_sparse_VB(args.depth, args.num_classes, lambda_n, prior_sigma_0, prior_sigma_1).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=0)
    learning_rate = args.init_lr

    PATH = args.base_path + args.model_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    num_epochs = args.nepoch
    train_accuracy_path = np.zeros(num_epochs)
    train_loss_path = np.zeros(num_epochs)

    test_accuracy_path = np.zeros(num_epochs)
    test_loss_path = np.zeros(num_epochs)
    sparsity_path = np.zeros(num_epochs)

    torch.manual_seed(args.seed)

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

    NTrain = len(train_loader.dataset)
    best_accuracy = 0

    for epoch in range(num_epochs):
        m.begin_epoch()
        net.train()
        epoch_training_loss = 0.0
        total_count = 0
        accuracy = 0

        if epoch in args.lr_decay_time:
            for para in optimizer.param_groups:
                para['lr'] = para['lr'] / args.lr_decay_factor
                learning_rate = para['lr']

        if epoch < args.lr_decay_time[0]:
            prior_sigma_0 = args.sigma1
            net.set_prior(lambda_n, prior_sigma_0, prior_sigma_1)
        else:
            prior_sigma_0 = args.sigma0
            net.set_prior(lambda_n, prior_sigma_0, prior_sigma_1)
        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            output = net(input)
            loss = loss_func(output, target)

            optimizer.zero_grad()

            loss.backward()

            with torch.no_grad():
                for para in net.parameters():
                    prior_grad = para.prior_grad.div(NTrain)
                    para.grad.data +=  prior_grad

            optimizer.step()

            epoch_training_loss += loss.mul(input.shape[0]).item()
            accuracy += output.data.argmax(1).eq(target.data).sum().item()
            total_count += input.shape[0]
            train_loss_path[epoch] = epoch_training_loss / total_count
            train_accuracy_path[epoch] = accuracy / total_count
        print("epoch: ", epoch, ", train loss: ", epoch_training_loss / total_count, "train accuracy: ",
              accuracy / total_count)

        # calculate test set accuracy
        with torch.no_grad():
            print('sigma0:', net.sigma_0, 'sigma1:', net.sigma_1, 'lambdan:', net.lambda_n)

            test_accuracy, test_loss = model_eval(args, net, test_loader, device, loss_func)
            test_loss_path[epoch] = test_loss
            test_accuracy_path[epoch] = test_accuracy
            print("epoch: ", epoch, ", test loss: ", test_loss, "test accuracy: ", test_accuracy)


            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), PATH + 'best_model.pt')

            print('best accuracy:', best_accuracy)

            param_overall = 0.
            flops_overall = 0.                    
            for name, module in net.named_modules():
                if 'downsample' not in name:
                    if isinstance(module, (VB_Conv2d)):
                        weight_sigma = torch.log1p(torch.exp(module.weight_rho))
                        weight_epsilon = torch.zeros_like(module.weight_mu).normal_()
                        weight = module.weight_mu + weight_sigma * weight_epsilon
                        if module.transposed:
                            conv_w = weight.permute(1, 0, 2, 3)   
                            conv_w = conv_w.view(-1, conv_w.shape[1]*conv_w.shape[2]*conv_w.shape[3]).T
                        else:
                            conv_w = weight
                            conv_w = conv_w.view(-1, conv_w.shape[1]*conv_w.shape[2]*conv_w.shape[3]).T
                        in_prune_channels = module.in_channels
                        if module.bias_mu is not None:
                            bias_sigma = torch.log1p(torch.exp(module.bias_rho))
                            bias_epsilon = torch.zeros_like(module.bias_mu).normal_()
                            bias = module.bias_mu + bias_sigma * bias_epsilon
                            arr1_l = torch.norm(torch.cat((conv_w,bias.expand(1, bias.size()[0])),0),1,0)
                            out_prune_channels = torch.sum((arr1_l!=0).float())
                            param_overall += (weight != 0).sum() + (bias != 0).sum()
                            flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1]+1)* \
                                            (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups
                        else: 
                            arr1_l = torch.norm(conv_w,1,0)
                            out_prune_channels = torch.sum((arr1_l!=0).float())
                            param_overall += (weight != 0).sum()
                            flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1])* \
                                            (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups                        
                    elif isinstance(module, (VB_Linear)):
                        weight_sigma = torch.log1p(torch.exp(module.weight_rho))
                        weight_epsilon = torch.zeros_like(module.weight_mu).normal_()
                        weight = module.weight_mu + weight_sigma * weight_epsilon
                        if module.bias_mu is not None:
                            bias_sigma = torch.log1p(torch.exp(module.bias_rho))
                            bias_epsilon = torch.zeros_like(module.bias_mu).normal_()
                            bias = module.bias_mu + bias_sigma * bias_epsilon
                            param_overall += (weight != 0).sum() + (bias != 0).sum()
                            flops_overall += (module.in_features+1)*module.out_features
                        else:
                            param_overall += (weight != 0).sum()
                            flops_overall += module.in_features*module.out_features
            flops_pruned_val = flops_overall.item()
            param_pruned_val = param_overall.item()
            sparsity_overall = param_pruned_val/total_num_para
            flops_ratio_val = flops_pruned_val/total_flops

        writer.add_scalar('data/loss_train', train_loss_path[epoch], epoch)
        writer.add_scalar('data/accuracy_train', train_accuracy_path[epoch], epoch)
        writer.add_scalar('data/loss_test', test_loss, epoch)
        writer.add_scalar('data/accuracy_test', test_accuracy, epoch)
        writer.add_scalar('data/sparsity_edge', sparsity_overall, epoch)
        writer.add_scalar('data/param_pruned', param_pruned_val, epoch)
        writer.add_scalar('data/flops_ratio', flops_ratio_val, epoch)
        writer.add_scalar('data/flops_pruned', flops_pruned_val, epoch)
        writer.add_scalar('data/learning_rate', learning_rate, epoch)

        print('Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Edge sparsity: {}, FLOPs ratio: {}, Param pruned: {}, FLOPs pruned: {}'.format(
                            epoch, train_loss_path[epoch], train_accuracy_path[epoch], test_loss, test_accuracy, 
                            sparsity_overall,flops_ratio_val,param_pruned_val,flops_pruned_val))

        m.end_epoch(epoch, train_loss_path[epoch], train_accuracy_path[epoch], test_loss, test_accuracy, 
                    sparsity_overall, param_pruned_val, flops_ratio_val, flops_pruned_val, learning_rate, args.batch_train)

        torch.save(net.state_dict(), PATH + 'last_model.pt')

    import pickle
    filename = PATH + 'result.txt'
    f = open(filename, 'wb')
    pickle.dump([train_loss_path, train_accuracy_path, test_loss_path, test_accuracy_path, sparsity_path], f)
    f.close()

    #-----------------fine tune-------------_#
    print('\n#-----------------fine tune results-------------#')
    PATH = args.base_path + args.model_path
    net.load_state_dict(torch.load(PATH + 'last_model.pt'))
    with torch.no_grad():
        test_accuracy, test_loss = model_eval(args, net, test_loader, device, loss_func)
    print("test loss: ", test_loss, "test accuracy: ", test_accuracy)


    with torch.no_grad():
        total_num_para_overall = 0
        epsilon = 1e-20
        for name, para in net.named_parameters():
            if 'mu' in name:
                total_num_para_overall += para.numel()
        ratio_array = torch.zeros(total_num_para_overall)
        count = 0
        for name, para in net.named_parameters():
            if 'mu' in name:
                temp_mu = para
            if 'rho' in name:
                temp_rho = para
                temp_sigma = torch.log(1 + torch.exp(para))
                temp_ratio = temp_mu.abs().div(temp_sigma + epsilon)
                size = para.numel()
                ratio_array[count:(count + size)] = temp_ratio.view(-1)
                count = count + size
        user_mask = {}
        target_ratio = args.prune_ratio
        threshold_index = int(np.floor(total_num_para_overall * (1 - target_ratio)))
        temp = ratio_array.sort().values
        ratio_threshold = temp[threshold_index]
        count = 0
        for name, para in net.named_parameters():
            if 'mu' in name:
                size = para.numel()
                mask = (ratio_array[count:(count + size)] < ratio_threshold).view(para.shape)
                user_mask[name] = mask
                count = count + size
            if 'rho' in name:
                user_mask[name] = mask
    net.set_prune(user_mask)
    with torch.no_grad():
        test_accuracy, test_loss = model_eval(args, net, test_loader, device, loss_func)
    print("test loss: ", test_loss, "test accuracy: ", test_accuracy)

    total_num_para_in = 0
    non_zero_element = 0
    total_num_para_overall = 0
    non_zero_element_overall = 0
    for name, para in net.named_parameters():
        if 'mu' in name:
            total_num_para_overall += para.numel()
            non_zero_element_overall += (para != 0).sum()
            if 'downsample' not in name and 'bn' not in name:
                total_num_para_in += para.numel()
                non_zero_element += (para != 0).sum()
    print('sparsity overall:', non_zero_element_overall.item() / total_num_para_overall)
    print('Total number of parameters overall:', total_num_para_overall)
    print('sparsity:', non_zero_element.item() / total_num_para_in)
    print('Total number of parameters:', total_num_para_in)

    param_overall = 0.
    flops_overall = 0.                    
    for name, module in net.named_modules():
        if 'downsample' not in name:
            if isinstance(module, (VB_Conv2d)):
                weight_sigma = torch.log1p(torch.exp(module.weight_rho))
                weight_epsilon = torch.zeros_like(module.weight_mu).normal_()
                weight = module.weight_mu + weight_sigma * weight_epsilon
                if module.transposed:
                    conv_w = weight.permute(1, 0, 2, 3)   
                    conv_w = conv_w.view(-1, conv_w.shape[1]*conv_w.shape[2]*conv_w.shape[3]).T
                else:
                    conv_w = weight
                    conv_w = conv_w.view(-1, conv_w.shape[1]*conv_w.shape[2]*conv_w.shape[3]).T
                in_prune_channels = module.in_channels
                if module.bias_mu is not None:
                    bias_sigma = torch.log1p(torch.exp(module.bias_rho))
                    bias_epsilon = torch.zeros_like(module.bias_mu).normal_()
                    bias = module.bias_mu + bias_sigma * bias_epsilon
                    arr1_l = torch.norm(torch.cat((conv_w,bias.expand(1, module.bias.size()[0])),0),1,0)
                    out_prune_channels = torch.sum((arr1_l!=0).float())
                    param_overall += (weight != 0).sum() + (bias != 0).sum()
                    flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1]+1)* \
                                    (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups
                else: 
                    arr1_l = torch.norm(conv_w,1,0)
                    out_prune_channels = torch.sum((arr1_l!=0).float())
                    param_overall += (weight != 0).sum()
                    flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1])* \
                                    (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups                        
            elif isinstance(module, (VB_Linear)):
                weight_sigma = torch.log1p(torch.exp(module.weight_rho))
                weight_epsilon = torch.zeros_like(module.weight_mu).normal_()
                weight = module.weight_mu + weight_sigma * weight_epsilon
                if module.bias_mu is not None:
                    bias_sigma = torch.log1p(torch.exp(module.bias_rho))
                    bias_epsilon = torch.zeros_like(module.bias_mu).normal_()
                    bias = module.bias_mu + bias_sigma * bias_epsilon
                    param_overall += (weight != 0).sum() + (bias != 0).sum()
                    flops_overall += (module.in_features+1)*module.out_features
                else:
                    param_overall += (weight != 0).sum()
                    flops_overall += module.in_features*module.out_features
    flops_pruned_val = flops_overall.item()
    param_pruned_val = param_overall.item()
    sparsity_overall = param_pruned_val/total_num_para
    flops_ratio_val = flops_pruned_val/total_flops
    print('Edge sparsity: ', sparsity_overall)
    print('Flop ratio: ', flops_ratio_val)

    PATH = args.base_path + args.model_path + args.fine_tune_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr_fine_tune, momentum=args.momentum_fine_tune, weight_decay=5e-4)

    num_epochs_fine_tune = args.nepoch_fine_tune
    train_accuracy_path_fine_tune = np.zeros(num_epochs_fine_tune)
    train_loss_path_fine_tune = np.zeros(num_epochs_fine_tune)

    test_accuracy_path_fine_tune = np.zeros(num_epochs_fine_tune)
    test_loss_path_fine_tune = np.zeros(num_epochs_fine_tune)

    sparsity_path_fine_tune = np.zeros(num_epochs_fine_tune)

    torch.manual_seed(args.seed)

    best_accuracy = 0

    prior_sigma_0 = args.sigma1
    net.set_prior(lambda_n, prior_sigma_0, prior_sigma_1)

    for epoch in range(num_epochs_fine_tune):
        m.begin_epoch()
        net.train()
        epoch_training_loss = 0.0
        total_count = 0
        accuracy = 0

        if epoch in args.lr_decay_time_fine_tune:
            for para in optimizer.param_groups:
                para['lr'] = para['lr']/args.lr_decay_factor
        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            output = net(input)
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            epoch_training_loss += loss.mul(input.shape[0]).item()
            accuracy += output.data.argmax(1).eq(target.data).sum().item()
            total_count += input.shape[0]
            train_loss_path_fine_tune[epoch] = epoch_training_loss / total_count
            train_accuracy_path_fine_tune[epoch] = accuracy / total_count
        print("epoch: ", epoch, ", train loss: ", epoch_training_loss / total_count, "train accuracy: ",
              accuracy / total_count)

        with torch.no_grad():

            test_accuracy, test_loss = model_eval(args, net, test_loader, device, loss_func)
            test_loss_path_fine_tune[epoch] = test_loss
            test_accuracy_path_fine_tune[epoch] = test_accuracy
            print("epoch: ", epoch, ", test loss: ", test_loss, "test accuracy: ", test_accuracy)

            total_num_para_in = 0
            non_zero_element = 0
            total_num_para_overall = 0
            non_zero_element_overall = 0
            for name, para in net.named_parameters():
                if 'mu' in name:
                    total_num_para_overall += para.numel()
                    non_zero_element_overall += (para != 0).sum()
                    if 'downsample' not in name and 'bn' not in name:
                        total_num_para_in += para.numel()
                        non_zero_element += (para != 0).sum()
            print('sparsity overall:', non_zero_element_overall.item() / total_num_para_overall)
            print('Total number of parameters overall:', total_num_para_overall)
            print('sparsity:', non_zero_element.item() / total_num_para_in)
            print('Total number of parameters:', total_num_para_in)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), PATH + 'best_model.pt')
            print('best accuracy:', best_accuracy)

            param_overall = 0.
            flops_overall = 0.                    
            for name, module in net.named_modules():
                if 'downsample' not in name:
                    if isinstance(module, (VB_Conv2d)):
                        weight_sigma = torch.log1p(torch.exp(module.weight_rho))
                        weight_epsilon = torch.zeros_like(module.weight_mu).normal_()
                        weight = module.weight_mu + weight_sigma * weight_epsilon
                        if module.transposed:
                            conv_w = weight.permute(1, 0, 2, 3)   
                            conv_w = conv_w.view(-1, conv_w.shape[1]*conv_w.shape[2]*conv_w.shape[3]).T
                        else:
                            conv_w = weight
                            conv_w = conv_w.view(-1, conv_w.shape[1]*conv_w.shape[2]*conv_w.shape[3]).T
                        in_prune_channels = module.in_channels
                        if module.bias_mu is not None:
                            bias_sigma = torch.log1p(torch.exp(module.bias_rho))
                            bias_epsilon = torch.zeros_like(module.bias_mu).normal_()
                            bias = module.bias_mu + bias_sigma * bias_epsilon
                            arr1_l = torch.norm(torch.cat((conv_w,bias.expand(1, bias.size()[0])),0),1,0)
                            out_prune_channels = torch.sum((arr1_l!=0).float())
                            param_overall += (weight != 0).sum() + (bias != 0).sum()
                            flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1]+1)* \
                                            (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups
                        else: 
                            arr1_l = torch.norm(conv_w,1,0)
                            out_prune_channels = torch.sum((arr1_l!=0).float())
                            param_overall += (weight != 0).sum()
                            flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1])* \
                                            (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]*(module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups                        
                    elif isinstance(module, (VB_Linear)):
                        weight_sigma = torch.log1p(torch.exp(module.weight_rho))
                        weight_epsilon = torch.zeros_like(module.weight_mu).normal_()
                        weight = module.weight_mu + weight_sigma * weight_epsilon
                        if module.bias_mu is not None:
                            bias_sigma = torch.log1p(torch.exp(module.bias_rho))
                            bias_epsilon = torch.zeros_like(module.bias_mu).normal_()
                            bias = module.bias_mu + bias_sigma * bias_epsilon
                            param_overall += (weight != 0).sum() + (bias != 0).sum()
                            flops_overall += (module.in_features+1)*module.out_features
                        else:
                            param_overall += (weight != 0).sum()
                            flops_overall += module.in_features*module.out_features
            flops_pruned_val = flops_overall.item()
            param_pruned_val = param_overall.item()
            sparsity_overall = param_pruned_val/total_num_para
            flops_ratio_val = flops_pruned_val/total_flops
            print('Edge sparsity: ', sparsity_overall)
            print('Flop ratio: ', flops_ratio_val)

        writer.add_scalar('data/loss_train', train_loss_path_fine_tune[epoch], epoch+num_epochs)
        writer.add_scalar('data/accuracy_train', train_accuracy_path_fine_tune[epoch], epoch+num_epochs)
        writer.add_scalar('data/loss_test', test_loss, epoch+num_epochs)
        writer.add_scalar('data/accuracy_test', test_accuracy, epoch+num_epochs)
        writer.add_scalar('data/sparsity_edge', sparsity_overall, epoch+num_epochs)
        writer.add_scalar('data/param_pruned', param_pruned_val, epoch+num_epochs)
        writer.add_scalar('data/flops_ratio', flops_ratio_val, epoch+num_epochs)
        writer.add_scalar('data/flops_pruned', flops_pruned_val, epoch+num_epochs)
        writer.add_scalar('data/learning_rate', learning_rate, epoch+num_epochs)

        print('Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Edge sparsity: {}, FLOPs ratio: {}, Param pruned: {}, FLOPs pruned: {}'.format(
                            epoch+num_epochs, train_loss_path_fine_tune[epoch], train_accuracy_path_fine_tune[epoch], test_loss, test_accuracy, 
                            sparsity_overall,flops_ratio_val,param_pruned_val,flops_pruned_val))

        m.end_epoch(epoch+num_epochs, train_loss_path_fine_tune[epoch], train_accuracy_path_fine_tune[epoch], test_loss, test_accuracy, 
                    sparsity_overall, param_pruned_val, flops_ratio_val, flops_pruned_val, learning_rate, args.batch_train)

        # torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')

    m.save(args.base_path + args.model_path,'results_Sun2021_VB_RN'+str(args.depth)+'_SGD_batch_'+str(args.batch_train)+'_prune_'+str(args.prune_ratio)+'_seed_'+str(args.seed))

    import pickle
    filename = PATH + 'result.txt'
    f = open(filename, 'wb')
    pickle.dump([train_loss_path_fine_tune, train_accuracy_path_fine_tune, test_loss_path_fine_tune,
                 test_accuracy_path_fine_tune, sparsity_path_fine_tune], f)
    f.close()

    writer.close()


if __name__ == '__main__':
    main()