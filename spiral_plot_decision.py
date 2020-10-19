import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from spiral import Cart2Polar, WrappedCart2Polar, PolarNet
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--net", default="polar", help="polar or polar_mod")
parser.add_argument("--hid", type=int, default=6)
parser.add_argument("--fc1", action="store_true")
parser.add_argument("--fc2", action="store_true")
args = parser.parse_args()

if args.net == 'polar':
    cart2polar = Cart2Polar()
elif args.net == 'polar_mod':
    cart2polar = WrappedCart2Polar()
else:
    print(f"error: unknown cart2polar type {args.net}")
    exit(1)

net = PolarNet(args.hid, cart2polar=cart2polar)

net_fname = f"spiral_{args.net}_{args.hid}.pt"
if os.path.exists(net_fname):
    print(f"info: loading model {net_fname}")
    net.load_state_dict(torch.load(net_fname))
else:
    print(f"error: couldn't load model {net_fname}")
    exit(1)

df = pd.read_csv('spirals.csv')
values = df.values

num_input = values.shape[1]-1

data = values[:, 0:num_input]
target = values[:, num_input:num_input+1]

data = cart2polar(torch.Tensor(data)).detach().numpy()

r = data[:,0]
theta = data[:,1]

target = torch.Tensor(target)

plt.scatter(r, theta, c=1-target[:,0], cmap='RdYlBu')

weight1 = net.fc1.weight.detach().numpy()
bias1 = net.fc1.bias.detach().numpy()
N1 = weight1.shape[0]

weight2 = net.fc2.weight.detach().numpy()
bias2 = net.fc2.bias.detach().numpy()

# plot linear boundaries
if args.fc1:
    theta_range = np.linspace(np.min(theta), np.max(theta), 5)


    for i in range(N1):
        w0, w1 = weight1[i,0:2]
        b0 = bias1[i]

        # w0x+w1y+b = 0
        # y = (-b-w0x)/w1
        r_range = (-b0-w1*theta_range)/w0
        plt.plot(r_range, theta_range)

if args.fc2:
    # plot decision boundary
    theta_range = np.linspace(np.min(theta)-10*np.pi, np.max(theta)+10*np.pi, 500)
    r_range = []
    pot_r_range = np.linspace(np.min(r)-3, np.max(r)+3, 500)

    theta_points = []

    for theta in theta_range:
        all_z1 = []
        for r in pot_r_range:
            x = np.mat([[r, theta]])
            z0 = weight1*x.transpose() + bias1.reshape(N1,1)
            z0 = np.tanh(z0)
            
            z1 = weight2*z0 + bias2.reshape(1,1)
            z1 = np.abs(z1)
            all_z1.append(z1[0,0])

        all_z1 = np.array(all_z1)
        mask = all_z1 < 0.1

        r_vals = pot_r_range[mask]
        r_range.extend(r_vals)
        theta_points.extend([theta for _ in range(len(r_vals))])

    plt.scatter(r_range, theta_points, c="orange", marker="x")
            

plt.xlabel("r")
plt.ylabel("rads")
plt.show()

