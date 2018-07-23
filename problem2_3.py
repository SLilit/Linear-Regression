import numpy as np
import sys

data = np.genfromtxt(sys.argv[1], delimiter = ",")
train = data[:, :2]
mean = train.mean(axis = 0)
std = train.std(axis = 0)
for element in train:
    for i in range(len(element)):
        element[i] = (element[i] - mean[i])/std[i]
bias = np.ones((data.shape[0],1))
X_train = np.concatenate((bias, train), axis = 1)
Y_train = data[:, 2]
new_weights = [0, 0, 0]
alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.0015]                       
w = []




