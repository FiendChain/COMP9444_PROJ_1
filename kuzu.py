# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

KUZU_SIZE = (28,28)
KUZU_TARGETS = 10

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.lin1 = nn.Linear(KUZU_SIZE[0]*KUZU_SIZE[1], KUZU_TARGETS)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        output = F.log_softmax(x, dim=1)
        return output

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        size = KUZU_SIZE[0]*KUZU_SIZE[1]
        layer_1_size = int(0.5*size)

        self.fc1 = nn.Linear(size, layer_1_size) 
        self.fc2 = nn.Linear(layer_1_size, KUZU_TARGETS)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
        

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self, activation=F.relu):
        super(NetConv, self).__init__()
        self.activation = activation

        conv1_ksize = 3
        conv1_stride = 1
        conv1_osize = math.ceil((KUZU_SIZE[0]-conv1_ksize)/conv1_stride) + 1

        conv2_ksize = 3
        conv2_stride = 1
        conv2_osize = math.ceil((conv1_osize-conv2_ksize)/conv2_stride + 1)

        max_pool_1 = 2
        max_pool_1_size = math.floor(conv2_osize/max_pool_1) 

        fc1_size = max_pool_1_size*max_pool_1_size*64

        self.conv1 = nn.Conv2d(1, 32, conv1_ksize, conv1_stride)
        self.conv2 = nn.Conv2d(32, 64, conv2_ksize, conv2_stride)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.maxpool1 = nn.MaxPool2d(max_pool_1)
        self.fc1 = nn.Linear(fc1_size, 128)
        self.fc2 = nn.Linear(128, KUZU_TARGETS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class NetConvMini(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self, activation=F.relu, dense_size=128, use_dropout=True):
        super(NetConvMini, self).__init__()
        self.use_dropout = use_dropout
        self.activation = activation
        self.dense_size = dense_size

        conv1_ksize = 5
        conv1_stride = 1
        conv1_osize = math.ceil((KUZU_SIZE[0]-conv1_ksize)/conv1_stride) + 1

        max_pool_1 = 2
        max_pool_1_size = math.floor(conv1_osize/max_pool_1) 

        conv2_ksize = 5
        conv2_stride = 1
        conv2_osize = math.ceil((max_pool_1_size-conv2_ksize)/conv2_stride + 1)

        max_pool_2 = 2
        max_pool_2_size = math.floor(conv2_osize/max_pool_2) 

        fc1_size = max_pool_2_size*max_pool_2_size*64

        self.conv1 = nn.Conv2d(1, 32, conv1_ksize, conv1_stride)
        self.conv2 = nn.Conv2d(32, 64, conv2_ksize, conv2_stride)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.maxpool1 = nn.MaxPool2d(max_pool_1)
        self.maxpool2 = nn.MaxPool2d(max_pool_2)
        self.fc1 = nn.Linear(fc1_size, dense_size)
        self.fc2 = nn.Linear(dense_size, KUZU_TARGETS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool2(x)

        if self.use_dropout:
            x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout2(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output