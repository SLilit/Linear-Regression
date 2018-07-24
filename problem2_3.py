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

def predict(train_row, weights):
    pre = 0
    for i in range(len(train_row)):
        pre += train_row[i]*weights[i]

    return pre


    
def linear_regression(train, label, weights, iters, alpha):
    bias = [0, 0, 0]
    n = len(train)
    loss1 = 1  
    while iters != 0:
        loss = 0
        for i in range(n):
            prediction = predict(train[i], weights)
            loss += (prediction - label[i])**2
            for j in range(len(bias)):
                bias[j] += (prediction - label[i])*train[i][j]
        loss = loss/(2*n)
        if loss < loss1:
            loss1 = loss
            for i in range(len(weights)):
                weights[i] -= alpha*bias[i]/n
        else:
            break
        
        iters -= 1
              
    return weights

for i in range(len(alpha)):
    update = linear_regression(X_train, Y_train, new_weights, 100, alpha[i])
    new_weights = update
    w.append([alpha[i], 100, new_weights[0], new_weights[1], new_weights[2]])
    

np.savetxt(sys.argv[2], w, delimiter = ",")


