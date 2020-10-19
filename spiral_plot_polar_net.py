from spiral import PolarNet, WrappedCart2Polar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import argparse
import os

from cmap import OrGr

# custom plotter for plotting all nodes in a polar network
# we instead ignore the cart2polar conversion, and check how the hidden
# layers separate the (r, theta) plane
def graph_polar_nodes(net, data, target):
    # get output of all the hidden nodes, and output ndoe
    with torch.no_grad(): 
        net.eval()       
        polar = net.cart2polar(data)
        r = polar[:,0]
        theta = polar[:,1]
    
    r_start, r_end = torch.min(r), torch.max(r)
    theta_start, theta_end = torch.min(theta), torch.max(theta)
    
    r_range = torch.arange(start=r_start, end=r_end, step=0.01, dtype=torch.float32)
    theta_range = torch.arange(start=theta_start, end=theta_end, step=0.01, dtype=torch.float32)
    R = r_range.repeat(theta_range.size()[0])
    T = torch.repeat_interleave(theta_range, r_range.size()[0], dim=0)
    grid = torch.cat((R.unsqueeze(1), T.unsqueeze(1)), 1)

    # plot output of the neuron
    # takes in the grid data
    def plot(data):
        plt.clf()
        plt.pcolormesh(
            r_range, theta_range,
            data.cpu().view(
                theta_range.size()[0],
                r_range.size()[0]), 
            cmap=OrGr)
        plt.colorbar()
        plt.scatter(r, theta,
                c=1-target[:,0], cmap='RdYlBu')
        plt.xlabel("r")
        plt.ylabel("rads")

    # graph the (r,theta) field 
    with torch.no_grad(): 
        net.eval()       
        x = net.fc1(grid)
        x = torch.tanh(x)

        hid1 = x
        hids = (hid1,)

        x = net.fc2(x)
        output = torch.sigmoid(x)

    # plot all hidden layers 
    for layer, hid in enumerate(hids):
        for node in range(hid.shape[1]):
            plot(hid[:,node])
            yield f"{layer}_{node}"

    plot(output)
    yield f"output"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net',type=str, default='polar',help='polar or polar_mod')
    parser.add_argument('--hid',type=int, default='10',help='number of hidden units')
    args = parser.parse_args()

    df = pd.read_csv("spirals.csv")
    values = df.values

    num_input = values.shape[1]-1
    data = values[:, 0:num_input]
    target = values[:, num_input:num_input+1]

    # choose network architecture
    if args.net == 'polar':
        net = PolarNet(args.hid)
    elif args.net == 'polar_mod':
        net = PolarNet(args.hid, WrappedCart2Polar())
    else:
        print(f'error: unknown polarnet type {args.net}')
        exit(1)

    net_prefix = f"spiral_{args.net}_{args.hid}"
    net_fname = f"{net_prefix}.pt"
    if os.path.exists(net_fname):
        net.load_state_dict(torch.load(net_fname))
    else:
        print(f'error: couldnt find polarnet {net_fname}')
        exit(1)

    plot_path = f"spiral_polplot_{args.net}_{args.hid}"
    os.makedirs(plot_path, exist_ok=True)
        
    for name in graph_polar_nodes(net, torch.Tensor(data), torch.Tensor(target)):
        plt.title(f"node_{name}")
        plt.savefig(f'{plot_path}/node_{name}.png')
        print(f"plotting: {name}    \r", end='')
        # plt.show()
    print()

if __name__ == '__main__':
    main()