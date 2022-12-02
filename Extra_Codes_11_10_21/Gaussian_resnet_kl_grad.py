import argparse

import torch
import numpy as np
import time
import torch.nn as nn
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
import adabound
from Gauss_resnet_models_kl_grad import Gauss_ResNet, Gauss_Wide_ResNet
from Gauss_layers_kl_grad import Gauss_layer
from Gauss_Conv_layers_kl_grad import Gauss_Conv2d_layer

parser = argparse.ArgumentParser(description='Cifar10 ResNet')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--results_path', default='./results_resnet/cifar10/Gauss/', type = str, help = 'base path for saving result')

# Resnet Architecture
parser.add_argument('-depth', default=20, type=int, help='depth of the resnet')
parser.add_argument('--widen_factor', default=10, type=int, help='width of the wide-resnet')

# Data setting
parser.add_argument('--num_classes', default = 10, type = int, help = 'total number of classes in classification dataset')

# Random Erasing
parser.add_argument('-p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('-sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('-r1', default=0.3, type=float, help='aspect of erasing area')

# Training Setting
parser.add_argument('--nepoch', default = 300, type = int, help = 'total number of training epochs')
parser.add_argument('--lr_decay_time', default = [150, 225], type = int, nargs= '+', help = 'when to multiply lr by 0.1')
parser.add_argument('--lr_decay_factor', default = 10, type = float, help = 'what factor to divide lr with')
parser.add_argument('--init_lr', default = 0.1, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
parser.add_argument('--batch_train', default = 128, type = int, help = 'batch size for training')
parser.add_argument('--num_MC_train', default = 1, type = int, help = 'Number of MC samples for training')
parser.add_argument('--num_MC_test', default = 5, type = int, help = 'Number of MC samples for testing')

# Prior Setting
parser.add_argument('--sigma0', default = 1, type = float, help = 'sigma_0^2 in prior')

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

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
#     lr = args.init_lr * (0.5 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_dim = args.num_classes
    num_MC_train = args.num_MC_train
    num_MC_test = args.num_MC_test

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

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_train, shuffle=True,num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    m = RunManager()
    m.begin_run()

    loss_func = nn.CrossEntropyLoss().to(device)

    net = Gauss_ResNet(depth=args.depth, num_classes=target_dim, sigma_0 = args.sigma0).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=0)
    # optimizer = adabound.AdaBound(net.parameters(), lr=1e-3, final_lr=0.1)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr)
    learning_rate = args.init_lr

    total_num_para = 0
    for module in net.modules():
        if isinstance(module, (Gauss_Conv2d_layer,Gauss_layer)):
            total_num_para += module.w_mu.numel()
            if module.v_mu is not None:
                total_num_para += module.v_mu.numel()

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
        # learning_rate = adjust_learning_rate(optimizer,epoch)

        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)

            nll_train = 0.
            outputs = torch.zeros(num_MC_train, labels.shape[0], target_dim).to(device)
            for it in range(num_MC_train):
                outputs[it] = net(images) 
                nll_train += loss_func(outputs[it], labels)
            output_mean = outputs.mean(dim=0)            
            loss = nll_train/float(num_MC_train)

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for para in net.parameters():
                    prior_grad = para.prior_grad.div(NTrain)
                    para.grad.data +=  prior_grad

            optimizer.step()

            train_loss += loss.mul(images.shape[0]).item()
            correct_train += output_mean.data.argmax(1).eq(labels.data).sum().item()

        with torch.no_grad():
            train_accuracy = correct_train / len(train_set)
            train_loss = train_loss / len(train_set)

            train_Loss.append(train_loss)
            train_Accuracy.append(train_accuracy)

            nll_test_comp = 0
            outputs = torch.zeros(num_MC_test, len(test_set), target_dim).to(device)
            final_labels = torch.empty(len(test_set)).to(device)   
            for cnt, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device) 
                final_labels[cnt*labels.shape[0]:(cnt+1)*labels.shape[0]] = labels               
                for it in range(num_MC_test):       
                    outputs[it,cnt*labels.shape[0]:(cnt+1)*labels.shape[0],:] = net(images)                    
                    nll_test_comp += loss_func(outputs[it,cnt*labels.shape[0]:(cnt+1)*labels.shape[0],:], labels).mul(images.shape[0])       
            test_loss = nll_test_comp/float(num_MC_test)
            output_mean = outputs.mean(dim=0)
            test_loss = test_loss / len(test_set)
            test_accuracy = output_mean.data.argmax(1).eq(labels.data).sum() / len(test_set)

            sparsity_overall = 0
            for module in net.modules():
                if isinstance(module, (Gauss_Conv2d_layer,Gauss_layer)):
                    sparsity_overall += (module.w != 0).sum()
                    if module.v is not None:
                        sparsity_overall += (module.v != 0).sum()
            sparsity_overall = sparsity_overall/total_num_para

            test_Loss.append(test_loss.item())
            test_Accuracy.append(test_accuracy.item())
            Edge_sparsity.append(sparsity_overall.item())

        writer.add_scalar('data/loss_train', train_loss, epoch)
        writer.add_scalar('data/accuracy_train', train_accuracy, epoch)
        writer.add_scalar('data/loss_test', test_loss.item(), epoch)
        writer.add_scalar('data/accuracy_test', test_accuracy.item(), epoch)
        writer.add_scalar('data/sparsity_edge', sparsity_overall.item(), epoch)
        writer.add_scalar('data/learning_rate', learning_rate, epoch)

        m.end_epoch(epoch, train_loss, train_accuracy,
                    test_loss.item(), test_accuracy.item(), sparsity_overall.item(), learning_rate, args.batch_train)

    print('Finished Training')
    writer.close()
    
    m.save(PATH,'results_Gaussian_Resnet_CIFAR10_silu_lr_decay_'+str(args.batch_train)) 

    torch.save(net.state_dict(), PATH + 'Gaussian_Resnet_CIFAR10_model_silu_lr_decay_'+ str(args.batch_train) + '.pt')

    plt.plot(range(num_epochs), train_Loss, 'b', label='Train')                
    plt.plot(range(num_epochs), test_Loss, 'orange', label='Test')
    plt.title('Gaussian: Train-Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.ylim([0.55, 1.0])
    plt.legend(loc='upper right', frameon=False)
    plt.grid(ls='dotted')
    plt.savefig(os.path.join(PATH,'Gaussian_Resnet_CIFAR10_loss_silu_lr_decay_'+str(args.batch_train)+'.png'),dpi=300)
    plt.close()

    plt.plot(range(num_epochs), train_Accuracy, 'b', label='Train')
    plt.plot(range(num_epochs), test_Accuracy, 'orange', label='Test')
    plt.title('Gaussian: Train-Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.ylim([0.55, 1.0])
    plt.legend(loc='upper right', frameon=False)
    plt.grid(ls='dotted')
    plt.savefig(os.path.join(PATH,'Gaussian_Resnet_CIFAR10_accuracy_silu_lr_decay_'+str(args.batch_train)+'.png'),dpi=300)
    plt.close()

    plt.plot(range(num_epochs), Edge_sparsity, 'g')
    plt.title('Gaussian: Edge Sparsity')
    plt.xlabel('Epochs')
    plt.ylabel('Sparsity')
    # plt.ylim([0.65, 1.0])
    plt.legend(loc='upper right', frameon=False)
    plt.grid(ls='dotted')
    plt.savefig(os.path.join(PATH,'Gaussian_Resnet_CIFAR10_sparsity_edge_silu_lr_decay_'+str(args.batch_train)+'.png'),dpi=300)
    plt.close()


if __name__ == '__main__':
    main()