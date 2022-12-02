import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import argparse
import errno

import os
from Gauss_linear_layers import SSGauss_Node_layer, Gauss_layer

parser = argparse.ArgumentParser(description='Simulation Regression')

# Basic Setting
parser.add_argument('--data_index', default=1, type = int, help = 'set data index')
parser.add_argument('--activation', default='tanh', type = str, help = 'set activation function')
args = parser.parse_args()

class my_Net_tanh(torch.nn.Module):
    def __init__(self):
        super(my_Net_tanh, self).__init__()
        self.fc1 = SSGauss_Node_layer(2000, 6)
        self.fc2 = SSGauss_Node_layer(6,4)
        self.fc3 = SSGauss_Node_layer(4,3)
        self.fc4 = Gauss_layer(3,1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class my_Net_relu(torch.nn.Module):
    def __init__(self):
        super(my_Net_relu, self).__init__()
        self.fc1 = SSGauss_Node_layer(2000, 6)
        self.fc2 = SSGauss_Node_layer(6,4)
        self.fc3 = SSGauss_Node_layer(4,3)
        self.fc4 = Gauss_layer(3,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def main():
    data_index = args.data_index
    subn = 500


    NTrain = 10000
    Nval = 1000
    NTest = 1000
    TotalP = 2000

    x_train = np.matrix(np.zeros([NTrain, TotalP]))
    y_train = np.matrix(np.zeros([NTrain, 1]))

    x_val = np.matrix(np.zeros([Nval, TotalP]))
    y_val = np.matrix(np.zeros([Nval, 1]))

    x_test = np.matrix(np.zeros([NTest, TotalP]))
    y_test = np.matrix(np.zeros([NTest, 1]))

    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/x_train.csv"))
    x_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/y_train.csv"))
    y_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/x_val.csv"))
    x_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/y_val.csv"))
    y_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/x_test.csv"))
    x_test[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/y_test.csv"))
    y_test[:, :] = temp[:, 1:]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    num_seed = 1

    num_selection_list = np.zeros([num_seed])
    num_selection_true_list = np.zeros([num_seed])
    train_loss_list = np.zeros([num_seed])
    val_loss_list = np.zeros([num_seed])
    test_loss_list = np.zeros([num_seed])

    for my_seed in range(num_seed):
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)

        if args.activation == 'tanh':
            net = my_Net_tanh()
        elif args.activation == 'relu':
            net = my_Net_relu()
        else:
            print('unrecognized activation function')
            exit(0)

        net.to(device)
        loss_func = nn.MSELoss()

        step_lr = 0.005
        optimization = torch.optim.SGD(net.parameters(), lr=step_lr, weight_decay=0)

        max_loop = 80001
        PATH = './result/regression/' + args.activation + '/DPF/'

        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        show_information = 100

        para_path = []
        for para in net.parameters():
            para_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))

        train_loss_path = np.zeros([max_loop // show_information + 1])
        val_loss_path = np.zeros([max_loop // show_information + 1])
        test_loss_path = np.zeros([max_loop // show_information + 1])

        for iter in range(max_loop):
            net.train()
            if subn == NTrain:
                subsample = range(NTrain)
            else:
                subsample = np.random.choice(range(NTrain), size=subn, replace=False)

            net.zero_grad()
            output = net(x_train[subsample,])
            nll_train = loss_func(output, y_train[subsample,])
            kl_train = 0.
            for module in net.modules():
                if isinstance(module, (SSGauss_Node_layer, Gauss_layer)):
                    kl_train += module.kl.div(subn)
            kl_train += net.kl_lambda.div(subn).squeeze()
            loss = nll_train + kl_train

            loss.backward()
            optimization.step()


            if iter % show_information == 0:
                net.eval()
                print('iteration:', iter)
                with torch.no_grad():
                    output = net(x_train)
                    loss = loss_func(output, y_train)
                    print("train loss:", loss)
                    train_loss_path[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_val)
                    loss = loss_func(output, y_val)
                    print("val loss:", loss)
                    val_loss_path[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_test)
                    loss = loss_func(output, y_test)
                    print("test loss:", loss)
                    test_loss_path[iter // show_information] = loss.cpu().data.numpy()

                    for i, para in enumerate(net.parameters()):
                        para_path[i][iter // show_information,] = para.cpu().data.numpy()


        import pickle

        filename = PATH + 'data_' + str(data_index) + "_simu_" + str(my_seed) + '_' + str(subn) + '_SS_Gauss.txt'
        f = open(filename, 'wb')
        pickle.dump([para_path, train_loss_path, val_loss_path, test_loss_path], f)
        f.close()

        with torch.no_grad():
            for i, para in enumerate(net.parameters()):
                para.data = torch.FloatTensor(para_path[i][-1,]).to(device)

        fine_tune_loop = 40001
        para_path_fine_tune = []

        for para in net.parameters():
            para_path_fine_tune.append(np.zeros([fine_tune_loop // show_information + 1] + list(para.shape)))

        train_loss_path_fine_tune = np.zeros([fine_tune_loop // show_information + 1])
        val_loss_path_fine_tune = np.zeros([fine_tune_loop // show_information + 1])
        test_loss_path_fine_tune = np.zeros([fine_tune_loop // show_information + 1])


        step_lr = 0.005
        optimization = torch.optim.SGD(net.parameters(), lr=step_lr)


        for iter in range(fine_tune_loop):
            net.train()
            if subn == NTrain:
                subsample = range(NTrain)
            else:
                subsample = np.random.choice(range(NTrain), size=subn, replace=False)

            net.zero_grad()
            output = net(x_train[subsample,])
            nll_train = loss_func(output, y_train[subsample,])
            kl_train = 0.
            for module in net.modules():
                if isinstance(module, (SSGauss_Node_layer, Gauss_layer)):
                    kl_train += module.kl.div(subn)
            kl_train += net.kl_lambda.div(subn).squeeze()
            loss = nll_train + kl_train

            loss.backward()
            optimization.step()


            if iter % show_information == 0:
                net.eval()
                print('iteration:', iter)
                with torch.no_grad():
                    output = net(x_train)
                    loss = loss_func(output, y_train)
                    print("train loss:", loss)
                    train_loss_path_fine_tune[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_val)
                    loss = loss_func(output, y_val)
                    print("val loss:", loss)
                    val_loss_path_fine_tune[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_test)
                    loss = loss_func(output, y_test)
                    print("test loss:", loss)
                    test_loss_path_fine_tune[iter // show_information] = loss.cpu().data.numpy()

                    for i, para in enumerate(net.parameters()):
                        para_path_fine_tune[i][iter // show_information,] = para.cpu().data.numpy()

        import pickle

        filename = PATH + 'data_' + str(data_index) + "_simu_" + str(my_seed) + '_' + str(subn) + '_' + '_SS_Gauss_fine_tune.txt'
        f = open(filename, 'wb')
        pickle.dump([para_path_fine_tune, train_loss_path_fine_tune, val_loss_path_fine_tune,
                     test_loss_path_fine_tune], f)
        f.close()

        output = net(x_train)
        loss = loss_func(output, y_train)
        print("Train Loss:", loss)
        train_loss_list[my_seed] = loss.cpu().data.numpy()

        output = net(x_val)
        loss = loss_func(output, y_val)
        print("Val Loss:", loss)
        val_loss_list[my_seed] = loss.cpu().data.numpy()

        output = net(x_test)
        loss = loss_func(output, y_test)
        print("Test Loss:", loss)
        test_loss_list[my_seed] = loss.cpu().data.numpy()

    import pickle

    filename = PATH + 'data_' + str(data_index) + '_result.txt'
    f = open(filename, 'wb')
    pickle.dump([num_selection_list,num_selection_true_list, train_loss_list, val_loss_list, test_loss_list], f)
    f.close()


if __name__ == '__main__':
    main()