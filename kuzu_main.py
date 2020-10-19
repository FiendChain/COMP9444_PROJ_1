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

from kuzu import NetLin, NetFull, NetConv

import os

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

    return (conf_matrix, accuracy) 

def main():
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--net',type=str,default='full',help='lin, full or conv')
    parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
    parser.add_argument('--mom',type=float,default=0.5,help='momentum')
    parser.add_argument('--epochs',type=int,default=10,help='number of training epochs')
    parser.add_argument('--no_cuda',action='store_true',default=False,help='disables CUDA')
    parser.add_argument('--save',action='store_true',default=False,help='enable saving')
    parser.add_argument('--infer',action="store_true",default=False,help='infer only')
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

    # choose network architecture
    if args.net == 'lin':
        net = NetLin().to(device)
    elif args.net == 'full':
        net = NetFull().to(device)
    elif args.net == 'conv':
        net = NetConv().to(device)
    else:
        print(f"error: unknown net type '{args.net}'")
        return 1

    summary(net, (1,28,28))
    if not args.save:
        print('warning: saving disabled')

    net_fname = f"kmnist_{args.net}.pt"
    if os.path.exists(net_fname):
        net.load_state_dict(torch.load(net_fname))

    if not list(net.parameters()):
        print(f"error: model {args.net} has no parameters")
        return 1

    if args.infer:
        conf_matrix, acc = test(args, net, device, test_loader)

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

    # training and testing loop
    try:
        # use SGD optimizer
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom)
        for epoch in range(1, args.epochs + 1):
            train(args, net, device, train_loader, optimizer, epoch)
            test(args, net, device, test_loader)
    except KeyboardInterrupt:
        pass
    finally:
        if args.save:
            print(f"info: saving model {net_fname}")
            torch.save(net.state_dict(), net_fname)
    
    return 0
        
if __name__ == '__main__':
    import sys
    sys.exit(main())
