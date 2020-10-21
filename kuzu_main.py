# kuzu_main.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sklearn.metrics as metrics
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import datasets, transforms
from torchsummary import summary

from kuzu import NetLin, NetFull, NetConv, NetConvMini

import os
import pickle

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    conf_matrix = np.zeros((10,10)) # initialize confusion matrix
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # determine index with maximal log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # update confusion matrix
            conf_matrix = conf_matrix + metrics.confusion_matrix(
                          target.cpu(),pred.cpu(),labels=[0,1,2,3,4,5,6,7,8,9])
        # print confusion matrix
        np.set_printoptions(precision=4, suppress=True)
        print(conf_matrix)
        print(f"{np.sum(conf_matrix)} samples tested")


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accuracy = correct / len(test_loader.dataset)

    return (conf_matrix, accuracy, test_loss) 

ACTIVATIONS = {
    "relu": torch.nn.functional.relu,
    "tanh": torch.nn.functional.tanh,
    "sigmoid": torch.nn.functional.sigmoid,
    "elu": torch.nn.ELU(),
    "selu": torch.nn.SELU(),
    "lelu": torch.nn.LeakyReLU()
}

ACTIVATION_KEYS = ','.join(ACTIVATIONS.keys())

def main():
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--net',type=str,default='full',help='lin, full, conv or conv_mini')
    parser.add_argument("--activation", type=str, default="relu", help=ACTIVATION_KEYS)
    parser.add_argument("--dense", type=int, default=128, help="number of dense layers in conv or conv_mini")
    parser.add_argument("--dropout", action="store_true", default=False, help="Apply dropout")
    parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
    parser.add_argument('--mom',type=float,default=0.5,help='momentum')
    parser.add_argument('--epochs',type=int,default=10,help='number of training epochs')
    parser.add_argument('--no_cuda',action='store_true',default=False,help='disables CUDA')
    parser.add_argument('--save',action='store_true',default=False,help='enable saving')
    parser.add_argument('--infer',action="store_true",default=False,help='infer only')
    parser.add_argument('--override', action='store_true', default=False, help="override existing save")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # fetch and load training data
    trainset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # fetch and load test data
    testset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    if args.activation in ACTIVATIONS:
        activation = ACTIVATIONS[args.activation]
    else:
        print(f"error: unknown activation function '{args.activation}'")
        return 1

    if args.dense < 0:
        print(f"error: negative number of dense neurons '{args.dense}'")
        return 1

    use_dropout_str = "dropout" if args.dropout else "no_dropout"

    # choose network architecture
    if args.net == 'lin':
        net = NetLin().to(device)
        model_prefix = 'lin'
    elif args.net == 'full':
        net = NetFull().to(device)
        model_prefix = 'full'
    elif args.net == 'conv':
        net = NetConv(activation=activation).to(device)
        model_prefix = f"conv_{args.activation}"
    elif args.net == 'conv_mini':
        net = NetConvMini(activation=activation, dense_size=args.dense, use_dropout=args.dropout).to(device)
        model_prefix = f"conv_mini_{args.activation}_{args.dense}_{use_dropout_str}"
    else:
        print(f"error: unknown net type '{args.net}'")
        return 1

    print(f"info: creating network {model_prefix}")

    summary(net, (1,28,28))
    if not args.save:
        print('warning: saving disabled')
    

    net_fname = f"kmnist_{model_prefix}.pt"
    log_fname = f"kmnist_{model_prefix}.log"

    if args.override:
        opt = input(f"overriding {net_fname} and {log_fname}. Are you sure? (y/n)")
        if opt.lower() != 'y':
            return 1

    logs = []

    if not args.override:
        if os.path.exists(log_fname):
            with open(log_fname, "rb") as fp:
                logs = pickle.load(fp)
                print(f"Resuming from log of size={len(logs)}")

        if os.path.exists(net_fname):
            net.load_state_dict(torch.load(net_fname))

    if not list(net.parameters()):
        print(f"error: model {args.net} has no parameters")
        return 1

    if args.infer:
        conf_matrix, acc, _ = test(args, net, device, test_loader)

        labels = "o,ki,su,tsu,na,ha,ma,ya,re,wo".split(",")
        df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
        plt.figure(figsize=(10,7))
        plt.title(f"Confusion matrix for {args.net} model")
        sn.heatmap(df_cm, annot=True, fmt="g")
        plt.ylabel("truth")
        plt.xlabel("predicted")
        plt.savefig(f"kuzu_{args.net}_conf.png")
        plt.show()

        return 0

    last_epoch = len(logs)+1

    # training and testing loop
    try:
        # use SGD optimizer
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom)
        for epoch in range(last_epoch, args.epochs + last_epoch):
            train(args, net, device, train_loader, optimizer, epoch)
            train_log = test(args, net, device, train_loader)
            test_log = test(args, net, device, test_loader)
            logs.append((train_log, test_log))

    except KeyboardInterrupt:
        pass
    finally:
        if args.save:
            print(f"info: saving model {net_fname}")
            torch.save(net.state_dict(), net_fname)
            with open(log_fname, "wb+") as fp:
                pickle.dump(logs, fp)
                print(f"saving logs size={len(logs)}")
    
    return 0
        
if __name__ == '__main__':
    import sys
    sys.exit(main())
