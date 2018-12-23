import requests
from sklearn.datasets import load_svmlight_file
from io import BytesIO
import time
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
X_train, X_test, Y_train, y_test = train_test_split(X_data, Y_data, test_size=0.25)# #random_state=2018)


# hyper parameter
learnning_rate = 0.0005
max_epoch = 200

np.random.seed(2018)
# initial parameter
# initialize with zeros
W = np.zeros((n_features+1, 1))
#b = 0


# W = np.random.normal(loc=0.0,scale=1.0,size=(n_features, 1))
# b = np.random.normal(loc=0.0, scale=1.0, size=(1, 1))
loss_train = []
loss_val = []

# gradient descent
for epoch in range(max_epoch):
    diff = np.matmul(X_train, W) - Y_train
    # calculate gradient 
    G_W =  (0.5 * W + np.matmul(X_train.transpose(), diff))
    # update parameter
    W -= learnning_rate * G_W

    y_predict = np.matmul(X_train, W)
    # loss= 0.5 * np.average(np.square(y_predict - Y_train))
    loss = np.average(np.abs(y_predict- Y_train))
    loss_train.append(loss)

    y_ = np.matmul(X_test, W)
    # loss = 0.5 * np.average(np.square(y_ - y_test))
    loss = np.average(np.abs(y_ - y_test))
    loss_val.append(loss)
    
# draw
import matplotlib.pyplot as plt 
# plt.figure(figsize=(16, 9))
plt.plot(loss_train, "-", color="r", label="train loss")
plt.plot(loss_val, "-", color="b", label="validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("The graph of absolute diff value varing with the number of iterations.")
plt.show()