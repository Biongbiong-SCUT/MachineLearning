# you can change the following hyper-parameter
penalty_factor = 0.5  # L2 regular term coefficients
learning_rate = 0.00005
max_epoch = 200
test_size = 0.25
batch_size = 128
# ------------------------------------------------------------------

# download the dataset
import requests
r = requests.get('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a')

# load the dataset
from sklearn.datasets import load_svmlight_file
from io import BytesIO
X, y = load_svmlight_file(f=BytesIO(r.content), n_features=123)

# preprocess the dataset
import numpy as np
n_samples, n_features = X.shape
X = X.toarray()
X = np.column_stack((X, np.ones((n_samples, 1))))
y = y.reshape((-1, 1))
# y = np.maximum(y, 0)
# devide the dataset into traning set and validation set
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)


print(X_train.shape)
print(y_train.shape)
W = np.random.normal(loc=0.0, scale=1.0, size=(n_features+1, 1))
for epoch in range(max_epoch):
    y_predict = np.matmul(X_train, W)
    test = y_train[y_predict * y_train >=1]
    