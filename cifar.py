import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Load the CIFAR-10 data
def load_CIFAR10_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR10_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR10_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def save_CIFAR10_images(X_train, y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        if not os.path.exists(cls):
            print("Make dir " + cls)
            os.makedirs(cls)
        for i, idx in enumerate(idxs):
            save_path = os.path.join(cls, cls + '_' + str(i) + '.jpg')
            plt.imsave(save_path, X_train[idx].astype('uint8'))

def save_CIFAR10_test_images(X_test, y_test):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_test == y)
        if not os.path.exists(cls):
            os.makedirs(cls)
        for i, idx in enumerate(idxs):
            save_path = os.path.join(cls, cls + '_' + str(i) + '.jpg')
            plt.imsave(save_path, X_test[idx].astype('uint8'))

# Load the data and save it
X_train, y_train, X_test, y_test = load_CIFAR10('./cifar-10-batches-py')
save_CIFAR10_test_images(X_test, y_test)
