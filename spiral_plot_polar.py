import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from spiral import Cart2Polar, WrappedCart2Polar
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--net", default="polar", help="polar or polar_mod")
args = parser.parse_args()

if args.net == 'polar':
    cart2polar = Cart2Polar()
elif args.net == 'polar_mod':
    cart2polar = WrappedCart2Polar()
else:
    print(f"error: unknown cart2polar type {args.net}")
    exit(1)

df = pd.read_csv('spirals.csv')
values = df.values

num_input = values.shape[1]-1

data = values[:, 0:num_input]
target = values[:, num_input:num_input+1]

data = cart2polar(torch.Tensor(data))

r = data[:,0]
theta = data[:,1]

target = torch.Tensor(target)

plt.scatter(r, theta, c=1-target[:,0], cmap='RdYlBu')
plt.xlabel("r")
plt.ylabel("rads")
plt.show()

