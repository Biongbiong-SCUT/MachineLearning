import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

dir_path = './ml-100k/'
'''
df_user = pd.read_csv(dir_path+'u.user', sep='|', names=['user_id' ,'age' , 'gender' , 'occupation', 'zip_code'])
df_item = pd.read_csv(dir_path+'u.item', sep='|', names=['movie id','movie title', 'release date', 'video release date',
              'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
              'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama','Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War','Western'], encoding='latin-1')

'''

header = ['user_id', 'item_id', 'rating', 'timestamp']
df_data = pd.read_csv(dir_path+'u.data', sep='\t', names=header)
df_train, df_test = train_test_split(df_data, test_size=0.25, random_state=2018)


df_train_data = df_train.set_index(['user_id','item_id'])['rating'].unstack().fillna(0)
df_test_data = df_test.set_index(['user_id','item_id'])['rating'].unstack().fillna(0)
df_score = df_data.set_index(['user_id','item_id'])['rating'].unstack().fillna(0)
tokens = tuple(zip(df_data['user_id'], df_data['item_id']))
tokens_train, tokens_test = train_test_split(tokens, test_size=0.25, random_state=2018)

test_mask = np.zeros(df_score.shape)
for token in tokens_test:
    i, j = token
    i, j = i-1, j-1
    test_mask[i, j] = 1

import random
def get_batch(token, batch_size):
    index = random.sample(range(len(token)),batch_size)
    batch = []
    for i in index:
        batch.append(token[i])
    return batch

#####################################################################################################
# the function simply update one sample
def svdCostAndGradient(userVector,itemVector, score):
    predicted = userVector.dot(itemVector.T)
    cost = 0.5 * np.square(score - predicted)
    delta = predicted - score
    gradUser = delta * (itemVector.flatten()) # (K,1)
    gradItem = delta * (userVector.flatten())
    return cost, gradUser, gradItem

# the fuction update a batch and return the total gradient 
def svdModel(userMatrix, itemMatrix, scoreMatrix, tokens, batch_size=1, normalize=False, trade_off=0.1):
    # initialize
    gradUser = np.zeros(userMatrix.shape)
    gradItem = np.zeros(itemMatrix.shape)
    cost = 0.0
    batch = get_batch(tokens, batch_size)
    for sample in batch:
        i, j = sample[0], sample[1]
        i, j = i-1, j-1
        uvec = userMatrix[i, :]
        ivec = itemMatrix[j, :]
        score = scoreMatrix[i, j]
        cc,gu,gi = svdCostAndGradient(uvec, ivec, score)
        gradUser[i] += gu 
        gradItem[j] += gi 
        cost += cc 
        if normalize:
            gradUser[i] += trade_off * uvec
            gradItem[j] += trade_off * ivec
            cost += trade_off * 0.5 * (np.sum(np.square(uvec)) + np.sum(np.square(ivec)))
    return cost, gradUser, gradItem

##################################################################################################
# helper function for test you can delete these utils funciton as you can
##################################################################################################
def sgd_wrapper(comVector, scoreMatrix, tokens, batch_size=1):
    U, V = scoreMatrix.shape
    userVector = comVector[:U, :]
    itemVector = comVector[U:U+V, :]
    cost, gradUser, gradItem = svdModel(userVector, itemVector, scoreMatrix, tokens, batch_size, normalize=True)
    comGrad = np.vstack((gradUser, gradItem))
    return cost, comGrad

def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost
      and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later
        ### YOUR CODE HERE:
        old_xix = x[ix]
        x[ix] = old_xix + h
        random.setstate(rndstate)
        fp = f(x)[0]
        x[ix] = old_xix - h
        random.setstate(rndstate)
        fm = f(x)[0]
        x[ix] = old_xix

        numgrad = (fp - fm)/(2* h)
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            return
    
        it.iternext() # Step to next dimension

    print("Gradient check passed!")

def test_Model():
    # sgd
    # initialize 
    scoreMatrix = df_score.values
    U,V = scoreMatrix.shape
    K = 15
    userMatrix = np.ones(shape=(U, K))
    itemMatrix = np.ones(shape=(V, K))
    tokens = tuple(zip(df_data['user_id'], df_data['item_id']))
    print("==== Gradient check for svd model ====")
    comVector = np.vstack((userMatrix, itemMatrix))
    gradcheck_naive(lambda vec: sgd_wrapper(vec, scoreMatrix, tokens, batch_size=10), comVector)
    return 

# just comment it if you don't need to debug
#test_Model()
###############################################################################################
def evalueate(userMatrix, itemMatrix, scoreMatrix, test_mask):
    loss = 0.0
    predictedMatrix = userMatrix.dot(itemMatrix.T)
    loss = np.sum(np.where(test_mask==1, np.square(predictedMatrix - scoreMatrix),  0))
    num = np.sum(test_mask)
    return loss / num



if __name__ == "__main__":
    
    loss = []
    val_loss = []

    scoreMatrix = df_score.values
    U,V = scoreMatrix.shape
    K = 15
    userMatrix = np.random.normal(loc=0.5, scale=0.1, size=(U, K))
    itemMatrix = np.random.normal(loc=0.5, scale=0.1, size=(V, K))
    #userMatrix = np.ones(shape=(U,K)) * 0.5
    #itemMatrix = np.ones(shape=(V,K)) * 0.5


    maxepoch = 800
    for epoch in range(maxepoch):
        batch_size = 1000
        # tips:
        # if batch_size is too small then the gained gradient is small as well 
        # therefore it require a larger learning rate to update the gradient
        # and trade_off should get smaller respect to the smaller batch_size
        # or it will cause some unexpected bias due to the regularization
        cost, gradUser, gradItem = svdModel(userMatrix, itemMatrix, scoreMatrix, tokens_train, 
            batch_size=batch_size, normalize=True, trade_off=0.1)
        # update gradient
        userMatrix -= 0.001 * gradUser
        itemMatrix -= 0.001 * gradItem
        if epoch % 5 == 0 :
            print(cost / batch_size)
            loss.append(cost / batch_size)
            test_loss = evalueate(userMatrix, itemMatrix, scoreMatrix, test_mask)
            print(test_loss)
            val_loss.append(test_loss)


    import matplotlib.pyplot as plt
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(loss, label='train loss')
    plt.plot(val_loss, label='test loss')
    plt.legend()
    plt.show()

