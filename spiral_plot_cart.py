import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

df = pd.read_csv('spirals.csv')
values = df.values

num_input = values.shape[1]-1

data = values[:, 0:num_input]
target = values[:, num_input:num_input+1]

x = data[:,0]
y = data[:,1]

plt.scatter(x, y, c=1-target[:,0], cmap='RdYlBu')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

