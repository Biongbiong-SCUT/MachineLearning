import numpy as np
np.random.seed(777)
# parameters
learning_rate = 0.1

x_data = np.asarray([
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1]])
y_data = np.asarray([
          [0],
          [1],
          [1],
          [0]])

# convert to column vector for engien vector
x_data = x_data.transpose()
y_data = y_data.transpose()

num_samples = 4.0

def relu(z):
    #return np.maximum(z, 0)
    return z * (z > 0)

def d_relu(z):
    return (z > 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

# define W and b
W1 = np.random.normal(0.0, 1.0, size=[2, 4])
b1 = np.random.normal(0.0, 1.0, size=[4])

W2 = np.random.normal(0.0, 1.0, size=[4, 4])
b2 = np.random.normal(0.0, 1.0, size=[4])

W3 = np.random.normal(0.0, 1.0, size=[4, 1])
b3 = np.random.normal(0.0, 1.0, size=[1])

def predict(X, W1, b1, W2, b2, W3, b3):
    # forwar pass
    z1 = np.matmul(W1.transpose(), X) + b1
    a1 = sigmoid(z1)

    z2 = np.matmul(W2.transpose(), a1) + b2
    a2 = sigmoid(z2)

    z3 = np.matmul(W3.transpose(), a2) + b3
    h = sigmoid(z3) 
    
    return h

def loss(y_predict, y_true):
    cross_entropy = -1 * np.average(y_true * np.log(y_predict) + 
            (1 - y_true) * np.log(1 - y_predict))
    return cross_entropy

def d_loss(y_predict, y_true):
    return -y_true/ y_predict + (1-y_true) / (1 - y_predict)

loss_val = []

for epoch in range(10001):
    # forward propagation
    # L1 input layer
    z1 = a1 = x_data
    
    # L2 hidden layer
    z2 = np.matmul(W1.transpose(), a1) + b1
    a2 = sigmoid(z2)

    # L2 hidden layer
    z3 = np.matmul(W2.transpose(), a2) + b2
    a3 = sigmoid(z3)

    # L4 ouput layer
    z4 = np.matmul(W3.transpose(), a3) + b3
    hypothesis = sigmoid(z4) 

    # backpropagation 
    # calculate error terms for each layer (i.e. dJ / dz)
    diff_l4 = d_loss(hypothesis, y_data) * d_sigmoid(z4)
    diff_l3 = np.matmul(W3, diff_l4) * d_sigmoid(z3)
    diff_l2 = np.matmul(W2, diff_l3) * d_sigmoid(z2) 

    # calculate partial derivatives
    partial_W3 = np.matmul(a3, diff_l4.transpose())
    partial_W2 = np.matmul(a2, diff_l3.transpose())
    partial_W1 = np.matmul(a1, diff_l2.transpose())

    # sum up all the samples
    partial_b3 = np.sum(diff_l4, axis=1)
    partial_b2 = np.sum(diff_l3, axis=1)
    partial_b1 = np.sum(diff_l2, axis=1)

    # update paremeter 
    W3 -= learning_rate * 1 / num_samples * partial_W3
    W2 -= learning_rate * 1 / num_samples * partial_W2
    W1 -= learning_rate * 1 / num_samples * partial_W1
    b3 -= learning_rate * 1 / num_samples * partial_b3
    b2 -= learning_rate * 1 / num_samples * partial_b2
    b1 -= learning_rate * 1 / num_samples * partial_b1

    loss_val.append(loss(hypothesis, y_data))
    if epoch % 200 == 0:
        print(hypothesis, loss(hypothesis, y_data))


y = predict(x_data,W1, b1, W2, b2, W3, b3)
correct = np.asarray(y > 0.5).astype('f')
accuracy = np.average(np.equal(correct, y_data).astype('f'))
print("\nHypothesis: ", y, "\nCorrect: ", correct, "\nAccuracy: ", accuracy)

# draw 
import matplotlib.pyplot as plt
plt.plot(loss_val, '-', color='r', label='train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("The entropy value varing with the number of iterations.")
plt.show()

'''
Hypothesis:  [[7.90367445e-04 9.97305696e-01 9.97901383e-01 2.51829106e-03]]
Correct:  [[0. 1. 1. 0.]]
Accuracy:  1.0
'''