import numpy as np
from engine.micrograd import *

from model import Mymodel
import h5py


def load_dataset():
    train_dataset = h5py.File('./dataset/train_catvnoncat.h5', 'r')
    test_dataset = h5py.File('./dataset/test_catvnoncat.h5', 'r')

    X_train = np.array(train_dataset['train_set_x'])
    Y_train = np.array(train_dataset['train_set_y']).reshape(1,-1)


    X_test = np.array(test_dataset['test_set_x'])
    Y_test = np.array(test_dataset['test_set_y']).reshape(1,-1)

    return X_train, Y_train, X_test, Y_test

X, Y, X_test, Y_test = load_dataset()

print("Train X ", X.shape)
print("Train Y ", Y.shape)
print("Test X ", X_test.shape)
print("Test Y ", Y_test.shape)

X_train = X.reshape(-1, X.shape[1]*X.shape[2]*X.shape[3]).T  # (m, w, h, rgb) -> (nx, m) 
X_train = X_train/255.
X_test = X_test.reshape(-1, X_test.shape[1]*X_test.shape[2]*X_test.shape[3]).T  # (m, w, h, rgb) -> (nx, m) 
X_test = X_test/255.
# print(X_train.shape)
nx = X_train.shape[0]


layer_dims = np.array([nx, 20, 7, 5, 1])

np.random.seed(1)
model = Mymodel(layer_dims)

for i in range(3000):
    # print(i)
    yh = model(X_train)
    
    loss = Value.BCE(Y, yh)
    # print(loss)
    loss.backward()
    model.optimize(0.0075)
    if (i % 100 == 0 or i == 3000 - 1):
        print("Cost after iteration {}: {}".format(i, loss.data))
    del loss, yh
    # if i % 100 == 0:
    #     costs.append(cost)

