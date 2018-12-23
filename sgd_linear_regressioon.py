import requests
from sklearn.datasets import load_svmlight_file
from io import BytesIO
import time
import random
# download dataset
data = requests.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale")

# load dataset

X_data, Y_data = load_svmlight_file(f=BytesIO(data.content), n_features=13)

import numpy as np

X_data = X_data.toarray()
n_samples, n_features = X_data.shape
# X_data = np.column_stack((X_data, np.ones((n_samples, 1))))
X_data = np.column_stack((X_data, np.ones((n_samples, 1))))
Y_data = Y_data.reshape(-1, 1)

# divide train and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, y_test = train_test_split(X_data, Y_data, 
                                    test_size=0.25, random_state=2018)

# hyper parameter
learnning_rate = 0.0005
max_epoch = 200
penalty_factor = 0.5  # L2 regular term coefficients
np.random.seed(2018)

def get_batch(X, Y, batch_size):
    index = [i for i in range(len(X))]
    index = random.sample(index, batch_size)
    return X[index], Y[index]

def train(max_epoch, batch_size):
    loss_train = []
    loss_val = []
    # initial parameter
    W = np.zeros((n_features+1, 1))
    #W = np.random.normal(loc=0.0,scale=1.0,size=(n_features+1, 1))
    # gradient descent
    for epoch in range(max_epoch):
        # 最大样本数位 379
        batch_x, batch_y = get_batch(X_train, Y_train, batch_size)

        diff = np.matmul(batch_x, W) - batch_y
        # calculate gradient 
        G_W =  (penalty_factor * W + np.matmul(batch_x.transpose(), diff))
        # update parameter
        W -= learnning_rate * G_W

        y_predict = np.matmul(batch_x, W)
        # loss= 0.5 * np.average(np.square(y_predict - Y_train))
        loss = np.average(np.abs(y_predict- batch_y))
        loss_train.append(loss)

        y_ = np.matmul(X_test, W)
        loss = 0.5 * np.average(np.square(y_ - y_test))
        loss = np.average(np.abs(y_ - y_test))
        loss_val.append(loss)
    return loss_train, loss_val

def draw(max_epoch, batch_size, label):
    loss_train, _= train(max_epoch, batch_size)
    if label == None:
        label = ("batch_size %d" % batch_size)
    plt.plot(loss_train, label=label)

def draw_loss(max_epoch, batch_size, label1, label2):
    loss_train, loss_val= train(max_epoch, batch_size)
    plt.plot(loss_train, label=label1)
    plt.plot(loss_val, label=label2)

# draw
import matplotlib.pyplot as plt 

#draw fig. 1. in the paper
#draw_loss(200, 379, "train loss", "validation loss")
draw(200, 50, None)
draw(200, 100, None)
draw(200, 200, None)
draw(200, 300, None)
draw(200, 379, 'total_num_of_data 379')
plt.legend()
plt.show()
'''
#draw fig. 3. in the paper
plt.xlabel("epoch")
plt.ylabel("loss")


loss_train, loss_val = train(200, 100)
plt.subplot(3,1,1)
plt.plot(loss_train, color='r', label='train loss, batch size %d' % 100)
plt.plot(loss_val,  label='valid loss, batch size %d' % 100)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
loss_train, loss_val = train(200, 200)
plt.subplot(3,1,2)
plt.plot(loss_train,color = 'r', label='train loss, batch size %d' % 200)
plt.plot(loss_val,  label='valid loss, batch size %d' % 200)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
loss_train, loss_val = train(200, 300)
plt.subplot(3,1,3)
plt.plot(loss_train,color='r', label='train loss, batch size %d' % 300)
plt.plot(loss_val, label='valid loss, batch size %d' % 300)
plt.legend()
plt.show()
'''