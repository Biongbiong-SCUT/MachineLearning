# you can change the following hyper-parameter
penalty_factor = 0.5  # L2 regular term coefficients
learning_rate = 0.00005
max_epoch = 2000
test_size = 0.25
batch_size = 512
# ------------------------------------------------------------------

# download the dataset
#import requests
#r = requests.get('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a')
# load the dataset
from sklearn.datasets import load_svmlight_file
from io import BytesIO
r = open('./a9a.txt', 'rb')

# load the dataset
from sklearn.datasets import load_svmlight_file
from io import BytesIO
X, y = load_svmlight_file(f=BytesIO(r.read()), n_features=123)

# preprocess the dataset
import numpy as np
n_samples, n_features = X.shape
X = X.toarray()
X = np.column_stack((X, np.ones((n_samples, 1))))
y = y.reshape((-1, 1))
y = np.maximum(y, 0)
# devide the dataset into traning set and validation set
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=136)


print(X_train.shape)
print(y_train.shape)

###############################################################################################
# Logistics Model
###############################################################################################

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def hypothesis(z):
    return sigmoid(z)

# define cross entropy as loss function
def loss(y_predict, y_true):
    cross_entropy = -1 * np.average(y_true * np.log(y_predict) + 
            (1 - y_true) * np.log(1 - y_predict))
    return cross_entropy

def d_loss(y_predict, y_true):
    return -y_true/ y_predict + (1-y_true) / (1 - y_predict)

import random
def get_batch(X, Y, batch_size):
    index = [i for i in range(len(X))]
    index = random.sample(index, batch_size)
    return X[index], Y[index]

def accuracy(y_predict, y_true):
    y_cast = np.where(y_predict > 0.5, 1, 0)
    return np.average(np.equal(y_cast, y_true))

###############################################################################################
# train
###############################################################################################
import time
t0 = time.time()
# initial parameter
W = np.zeros((n_features+1, 1))
losses_val = []
losses_train = []
acc_log = []
epochs = []
for epoch in range(max_epoch):
    # get batches
    batch_x, batch_y = get_batch(X_train, y_train, batch_size)
    
    # forward inference
    z = np.matmul(batch_x, W)
    y = hypothesis(z)

    # calulate gradient
    # dl / dy * dy / dz * dz/ dw
    diff_loss = d_loss(y, batch_y) #(, 1)
    diff_y = diff_loss * d_sigmoid(z) # (, 1)
    diff_z = batch_x #(, 124)
    partial_w = np.matmul(diff_z.transpose(), diff_y)

    # update parameter
    W -= partial_w * learning_rate

    if epoch % 20 == 0:
        epochs.append(epoch)
        # train loss
        y_predict = hypothesis(np.matmul(batch_x, W))
        loss_train = np.average(loss(y_predict, batch_y))
        losses_train.append(loss_train)
        #
        # validation loss
        y_predict = hypothesis(np.matmul(X_val, W))
        loss_val = np.average(loss(y_predict, y_val))
        losses_val.append(loss_val) 
        acc = accuracy(y_predict, y_val)
        acc_log.append(acc)
        print("accuracy", acc)
        print(loss_val)

print('训练用时{}秒'.format(time.time() - t0))
# draw 
import matplotlib.pyplot as plt
# draw loss
plt.figure(1)
plt.plot(epochs, losses_train, "-", color='c', label="train loss")
plt.plot(epochs, losses_val, "-", color='r', label="validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss of logistic regression for a9a Data set")
plt.legend()

# draw accuracy 
plt.figure(2)
plt.plot(epochs, acc_log, 'bx-', label='accuracy')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Accuracy of logistic regression for a9a Data set")
plt.show()