# -*- coding: utf-8 -*-
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_svmlight_file

X_train, y_train = load_svmlight_file("a9a", n_features=123)
X_train = X_train.todense()

X_train = np.c_[X_train, np.ones((X_train.shape[0], 1))]

# print(y_train)
y_train = y_train.reshape((-1, 1))
# print(y_train)
y_train = np.where(y_train < 0, 0, 1)
# print(y_train)

X_val, y_val = load_svmlight_file("a9a.t", n_features=123)
print(X_val.shape)
print(y_val.shape)
X_val = X_val.todense()
X_val = np.c_[X_val, np.ones((X_val.shape[0], 1))]

# print(X_train.shape)
# print(X_val.shape)

y_val = y_val.reshape((-1, 1))
y_val = np.where(y_val < 0, 0, 1)

# w(124,1）
# x(128,124)
# y(128,1)

# W = np.random.randn(X_train.shape[1], 1)
# W = np.random.randn(X_train.shape[1], 1)*0.2 2000
W = np.random.random((124, 1)) * 0.2
eta = 0.001
batch_size = 256
n_epoch = 2000
loss = []


# 124,1

def gx(yi, xi, w):
    wx = xi.dot(w)
    y = yi.T.dot(wx)
    y2 = xi.T.dot(yi)
    y_pred = 1 - y
    y2 = y2 * -1
    if (y_pred < 0):
        return 0
    else:
        return y2


def max(yi, xi, w):
    wx = xi.dot(w)
    y = yi.T.dot(wx)
    y_pred = 1 - y
    if (y_pred < 0):
        return 0
    else:
        return y


for epoch in range(n_epoch):
    beg = np.random.randint(X_train.shape[0] - batch_size)
    Xi = X_train[beg:(beg + batch_size)]
    yi = y_train[beg:(beg + batch_size)]
    cx = gx(yi, Xi, W)
    C = 0.1
    cx = cx * C
    G = W + cx
    W -= eta * G
    m = max(yi, Xi, W)
    a = np.linalg.norm(W)
    a = a * a
    Lvalidation = 0.5 * a + m * C
    loss.append(Lvalidation)

plt.figure(figsize=(18, 6))
plt.plot(loss,label="loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("The graph of absolute diff value varing with the number of iterations")
plt.show()