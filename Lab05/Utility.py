# -*- coding: utf-8 -*-

"""

Steven Nuñez Murillo - B95614
10-2022
Machine Learning II-2022

"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

# Custom subdirectory to find images
DIRECTORY = "images"

def load_data():
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    names = [n.decode('utf-8') for n in unpickle(DIRECTORY+"/batches.meta")[b'label_names']]
    x_train = None
    y_train = []
    for i in range(1,6):
        data = unpickle(DIRECTORY+"/data_batch_"+str(i))
        if i>1:
            x_train = np.append(x_train, data[b'data'], axis=0)
        else:
            x_train = data[b'data']
        y_train += data[b'labels']
    data = unpickle(DIRECTORY+"/test_batch")
    x_test = data[b'data']
    y_test = data[b'labels']

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    
    x_train = x_train.view( 50000,3,32,32)
    y_train = y_train.view(50000, 1)
    x_test = x_test.view(10000, 3, 32, 32)
    y_test = y_test.view(10000, 1)

    
    return names,x_train,y_train,x_test,y_test

names,x_train,y_train,x_test,y_test = load_data()

def plot_tensor(tensor, perm=None):
    if perm==None: perm = (1,2,0)
    plt.figure()
    plt.imshow(tensor.permute(perm).numpy().astype(np.uint8))
    plt.show()