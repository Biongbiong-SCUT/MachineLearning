# you can change the following hyper-parameter
penalty_factor = 0.5  # L2 regular term coefficients
learning_rate = 0.00005
max_epoch = 500
test_size = 0.25
batch_size = 512
# -------------------------------------------------------------------------------------------

# it's optional get dataset online 
# download the dataset
# import requests
# r = requests.get('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a')
# load the dataset 
from sklearn.datasets import load_svmlight_file
from io import BytesIO
r = open('./a9a.txt', 'rb')

X, y = load_svmlight_file(f=BytesIO(r.read()), n_features=123)

# preprocess the dataset
import numpy as np
n_samples, n_features = X.shape
X = X.toarray()
X = np.column_stack((X, np.ones((n_samples, 1))))
y = y.reshape((-1, 1))

# devide the dataset into traning set and validation set
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=136)


import random
def get_batch(X, Y, batch_size):
    index = [i for i in range(len(X))]
    index = random.sample(index, batch_size)
    return X[index], Y[index]

###############################################################################################
# SVM Model
###############################################################################################

def loss(y_predict, y_true):
    return np.maximum(0, 1 - y_true * y_predict)

def d_loss(X, W, y_true):
    G_w = np.zeros(W.shape)
    for (x_i, y_i) in zip(X, y_true):
        x_i.reshape(1, x_i.shape[0])
        z = np.matmul(x_i, W)
        if 1 - y_i * z > 0:
            G_w += (-y_i * x_i).reshape(W.shape)
    return G_w

def accuracy(y_predict, y_true):
    hinge_loss = loss(y_predict, y_true)
    return np.average(np.where(hinge_loss < 1, 1, 0))

###############################################################################################
# train
###############################################################################################
import time

t0 = time.time()
# initialize 
W = np.random.normal(loc=0, scale=1, size=(n_features + 1, 1))
# W = np.zeros(shape=(n_features+1, 1))
print('start...')
losses_train = []
losses_val = []
# object loss which require the regularize term
# result show that the objective loss is strict descenting
# while the train loss is trembling 

loss_objective = []
acc_log = []
epochs = []

for epoch in range(max_epoch):
    # compute gradient and update parameters
    batch_x, batch_y = X_train, y_train # full data
    # batch_x, batch_y = get_batch(X_train, y_train, batch_size) 
    G_w = W + d_loss(batch_x, W, batch_y)
    W -= learning_rate * G_w

    # print loss
    y_predict = np.matmul(batch_x, W)
    loss_train = np.average(loss(y_predict, batch_y))
    loss_val = np.average(loss(np.matmul(X_val, W), y_val))
    if epoch % 1 == 0:
        losses_train.append(loss_train)
        losses_val.append(loss_val)
        epochs.append(epoch)
        acc = accuracy(np.matmul(X_val, W), y_val)
        acc_log.append(acc)
        # regularization loss
        # l2 = 0.5 * np.sqrt(np.sum(np.square(W))) **2
        l2 = 0.5 * np.linalg.norm(W)
        loss_objective.append(l2)
        print(loss_train)
        print('accuracy: ', acc)
    

print('训练用时{}秒'.format(time.time() - t0))
# draw 
import matplotlib.pyplot as plt
# draw train loss and validation loss
plt.figure(1)
plt.plot(epochs, losses_train, "-", color='c', label="train loss")
plt.plot(epochs, losses_val, "-", color='r', label="validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss of support vector mechine for a9a Data set")
plt.legend()

# draw objective loss

plt.figure(2)
plt.plot(epochs, loss_objective, "-", color='c', label="objective loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Objective loss of support vector mechine for a9a Data set")
plt.legend()

# draw accuracy 
plt.figure(3)
plt.plot(epochs, acc_log, 'b-', label='accuracy')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Accuracy of support vector mechine for a9a Data set")
plt.show()

