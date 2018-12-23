import numpy as np
# hyper parameter
learning_rate = 0.01
np.random.seed(777) # for reproducibility

# Training Data
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
m = train_X.shape[0] # m denotes numbers of samples
# convert to column vector 
X_train = train_X.transpose()
Y_train = train_Y.transpose()

# forward pass
# initial parameter
# W = np.random.normal(0.0, 1.0)  
# b = np.random.normal(0.0, 1.0)
# in order to check our model, we set the initial 
# value as the version of tf
W = 2.2086694
b = -0.8204183

# define our model y = Wx + b
def hypothesis(W, b):
    return X_train * W + b

# define our loss function RMSE
def loss(W, b):
    return 1 / (2 * float(m)) * np.dot(W * X_train + b - Y_train, W * X_train + b - Y_train)

# for output
loss_val = []

for epoch in range(1000):
    # compute error term
    diff = hypothesis(W, b) - Y_train

    # compute gradient for weigths and bias
    G_w = 1 / float(m) * np.matmul(diff.transpose(), X_train)
    G_b = np.average(diff)
    
    # update paremeters
    b -= learning_rate * G_b
    W -= learning_rate * G_w
    
    if epoch % 200 == 0:
        print(epoch, loss(W, b), W, b)
    
    loss_val.append(loss(W, b))

print("W = ", W, " b =", b, " loss = ", loss(W, b))


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, train_X * W + b, label='Fitted line')
plt.xlabel("x")
plt.ylabel("y")
plt.figure(2)
plt.plot(loss_val)
plt.show()

'''
200 0.20810790483923886 0.45697420361829144 -0.6569617006050175
400 0.1576287042242674 0.41269065587310166 -0.34301127883658994
600 0.1265744754215395 0.377957315951072 -0.09676753204618373
800 0.10747026785122075 0.35071458216919466 0.09637118868502578
W=:  0.3294414711711726  b= 0.24718797076818763  loss =  0.09576327293101893
'''