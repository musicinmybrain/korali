import numpy as np
import pickle
import os
from subprocess import check_output

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(cifar_dir):
    """ load all of cifar """
    xs = []
    ys = []
    # Load all batches
    for b in range(1,6):
        f = os.path.join(cifar_dir, "cifar-10-batches-py", f'data_batch_{b}')
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(cifar_dir, "cifar-10-batches-py", 'test_batch'))
    return Xtr, Ytr, Xte, Yte

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_CIFAR10("/home/pollakg/polybox/CSE/master/6th_term/master_thesis/korali/data/Cifar10/cifar-10-batches-py")

    # N, 32 x 32 x 3 channels
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
