import requests
from sklearn.datasets import load_svmlight_file
from io import BytesIO

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
X_train, X_test, Y_train, y_test = train_test_split(X_data, Y_data, test_size=0.25, random_state=2018)


# W = np.zeros((n_features+1, 1))
mat_x = np.asmatrix(X_train)
W = np.matmul(np.matmul(mat_x.transpose(), mat_x).I, mat_x.transpose())
W = np.matmul(W, Y_train)

y_predict = np.matmul(W.transpose(), X_test.transpose()).transpose()
validation_loss = np.mean(np.square(y_predict - y_test))
print(validation_loss)