# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np


# cartesion to polar module
class Cart2Polar(torch.nn.Module):
    def __init__(self):
        super(Cart2Polar, self).__init__()
    
    def forward(self, X):
        x = X[:,0]
        y = X[:,1]
        r = torch.sqrt(x*x + y*y)
        a = torch.atan2(y,x)

        output = torch.stack([r,a], dim=0)
        output = torch.transpose(output, 0, 1)
        return output

# cartesion to polar with unwrapping of angle
class WrappedCart2Polar(torch.nn.Module):
    def __init__(self, r_mod=2, r_off=-1.5):
        super(WrappedCart2Polar, self).__init__()
        self.r_mod = r_mod 
        self.r_off = r_off
    
    def forward(self, X):
        x = X[:,0]
        y = X[:,1]
        r = torch.sqrt(x*x + y*y)
        a = torch.atan2(y,x)

        # make data even easier to separate
        k = torch.ceil((r+self.r_off)/self.r_mod)
        a = a - 2*math.pi*k

        output = torch.stack([r,a], dim=0)
        output = torch.transpose(output, 0, 1)
        return output

# two fully connected layers with tanh and 
# single fully connected output with sigmoid
class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(num_hid, num_hid)
        self.fc3 = nn.Linear(num_hid, 1)

        self.hids = []

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        hid1 = x

        x = self.fc2(x)
        x = torch.tanh(x)
        hid2 = x
        self.hids = (hid1, hid2)

        x = self.fc3(x)
        output = torch.sigmoid(x)
        return output

# convert cartesian to polar form
# single fully connected layer with tanh
# single fully connected output layer with sigmoid
class PolarNet(torch.nn.Module):
    def __init__(self, num_hid, cart2polar=None):
        super(PolarNet, self).__init__()
        if cart2polar is None:
            cart2polar = Cart2Polar()
        self.cart2polar = cart2polar
        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(num_hid, 1)

        self.hids = []

    def forward(self, x):
        x = self.cart2polar(x)
        x = self.fc1(x)
        x = torch.tanh(x)

        hid1 = x
        self.hids = (hid1,)

        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output

# choose a nicer colur for visualisation
top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Greens', 128)

newcolors = np.vstack((top(np.linspace(0.5, 1.0, 128)),
                       bottom(np.linspace(0, 0.5, 128))))
OrGr = ListedColormap(newcolors, name='OrGr')
    
# plot based on specified layer and node
# based on project specifications
def graph_hidden(net, layer, node):
    x_range = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    y_range = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    X = x_range.repeat(y_range.size()[0])
    Y = torch.repeat_interleave(y_range, x_range.size()[0], dim=0)
    points = torch.cat((X.unsqueeze(1), Y.unsqueeze(1)), 1)

    with torch.no_grad():
        net.eval()
        output = net.forward(points)
        hids = net.hids

    # layer starts from 1 
    hid = hids[layer-1]
    data = hid[:,node]

    plt.clf()
    plt.pcolormesh(
        x_range, y_range,
        data.cpu().view(
            y_range.size()[0],
            x_range.size()[0]), 
        cmap=OrGr)
    plt.colorbar()

# custom plotter for plotting all nodes in network
def graph_all_nodes(net):
    x_range = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    y_range = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    X = x_range.repeat(y_range.size()[0])
    Y = torch.repeat_interleave(y_range, x_range.size()[0], dim=0)
    points = torch.cat((X.unsqueeze(1), Y.unsqueeze(1)), 1)

    # plot output of the neuron
    def plot(data):
        plt.clf()
        plt.pcolormesh(
            x_range, y_range,
            data.cpu().view(
                y_range.size()[0],
                x_range.size()[0]), 
            cmap=OrGr)
        plt.colorbar()

    # get output of all the hidden nodes, and output ndoe
    with torch.no_grad(): 
        net.eval()       
        output = net.forward(points)
        hids = net.hids

    for layer, hid in enumerate(hids):
        for node in range(hid.shape[1]):
            plot(hid[:,node])
            yield f"{layer}_{node}"

    plot(output)
    yield f"output"
    