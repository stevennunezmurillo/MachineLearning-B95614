from msilib.schema import ServiceInstall
from tkinter import Y
from xml.sax import SAXNotSupportedException
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import random

def MSE(y_true, y_predict):
    
    error = 0
    
    for index in range(len(y_true)):
        error += ((y_true[index] - y_predict[index])**2)
    
    error *= 1/ len(y_true)
    
    #mse = mean_squared_error(y_true, y_predict)
    
  
def score(y_true, y_predict):
    
    y_true.to_numpy()
    y_predict.to_numpy()
    
    first_sum = sum(((y_true - y_predict)**2))
    mean = y_true.mean()
    second_sum = sum(((y_true - mean)**2))
    
    score_r2 = (second_sum-first_sum)/second_sum
    #r2 = r2_score(y_true, y_predict) 
        
    print(score_r2)

def fit(self, x, y, max_epochs=100, threshold=0.01, learning_rate=0.001, momentum=0, decay=0, error='mse', regularization='none', lambdas = 0):
    coeficients = []
    epoch = 0
    error_change = []
    
    df.insert(0, "ones", 1, allow_duplicates=False)
    
    data_x = x.to_numpy()
    ones = np.ones(np.shape(data_x)[0], 1)
    x_ones = np.concatenate((ones, data_x), axis=0)
    
    
    for elements in range(len(y)):
        coeficient = random()
        if coeficient not in coeficients:
            coeficients.append(coeficient)
    
    while epoch < max_epochs:
        if epoch != 0:
            if (error_change[epoch]-error_change[epoch-1] > threshold):
                break
            else:
                pass
    
    epoch+=1
        
    



Y_true = pd.Series([1,2,3,4,5])
  
Y_pred = pd.Series([0.6,1.29,1.99,2.69,3.4])

MSE(Y_true, Y_pred)
score(Y_true, Y_pred)

    
data = pd.read_csv("../data/fish_perch.csv")
print(data)

data.insert(0, "ones", 1, allow_duplicates=False)
print(data)
