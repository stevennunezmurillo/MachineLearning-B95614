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
        error += ((y_predict[index] - y_true[index])**2)
    
    error *= 1/ len(y_true)
    
    return error
    #mse = mean_squared_error(y_true, y_predict)
    
  
def score(y_true, y_predict):
    
    y_true.to_numpy()
    y_predict.to_numpy()
    
    first_sum = sum(((y_true - y_predict)**2))
    mean = y_true.mean()
    second_sum = sum(((y_true - mean)**2))
    
    score_r2 = (second_sum-first_sum)/second_sum
    #r2 = r2_score(y_true, y_predict) 
        
    #print(score_r2)

def fit(x, y, max_epochs=100, threshold=0.01, learning_rate=0.001, momentum=0, decay=0, error='mse', regularization='none', lambdas = 0):
    
    #Se agrega el bias  
    x.insert(0, "ones", 1, allow_duplicates=False)

    #Se pasa de dataframe a matriz numpy
    x_matrix = x.to_numpy()
    coeficients = np.array([random.sample(range(len(x_matrix[0])),len(x_matrix[0]))])
    epoch = 0
    epoch_error = []
    
    coeficients = np.transpose(coeficients)
    
    #Algoritmo de RG
    while epoch < max_epochs:

        xc_pred = xc_matrix_gen(x_matrix, coeficients)
        epoch_error.append(MSE(y, xc_pred))
        print(epoch_error[epoch])
        """
        if epoch != 0:
            print(epoch_error[epoch])
            #print(str(epoch_error[epoch]-epoch_error[epoch-1]) + ">" + str(threshold))
            if (epoch_error[epoch]-epoch_error[epoch-1] <= threshold):
                return coeficients
        """
        change = derivatives_errorc(xc_pred, y, x_matrix)
        coeficients = new_coeficients(coeficients, change, learning_rate)
        learning_rate = learning_rate/(1+decay)
        epoch+=1

    return coeficients
        

def xc_matrix_gen(x, c):
    """
    xc = []
    
    for row in range(len(x)):
        xc.append(sum(x[row]*c))
    """
    return np.dot(x,c)
           

def error_rg(xc_predict, y_true):
    return ((1/len(y_true)*sum((xc_predict-y_true)**2)))
            

def derivatives_errorc(xc_predict, y_true, x_ones):

    """
    error_change = []

    first_term = (2/len(y_true))*sum(xc_predict-y_true)

    sum_columns = [sum(x) for x in zip(*x_ones)]

    for i in range(len(sum_columns)):
        error_change.append(first_term * sum_columns[i])
    """
    #print(xc_predict.shape)
    #print(y_true.shape)
    #print(x_ones.shape)
    #(2/len(xc_predict))*((xc_predict-y_true).transpose()*x_ones).transpose()
    return (2/len(xc_predict))*np.dot((xc_predict-y_true).transpose(),x_ones).transpose()

def new_coeficients(coeficients, error_rg, eta):
    """
    print(coeficients)
    print("-----------------")
    print(eta)
    print("-----------------")
    print(error_rg)
    """

    return coeficients - (error_rg *eta)

def main():

    x = pd.read_csv("../data/fish_perch.csv")
    y = np.array([x['Weight']]).transpose()
    x.drop(['Weight'], axis=1, inplace = True)

    print(fit(x, y))


if __name__ == "__main__":
    main()
    
"""
Y_true = pd.Series([1,2,3,4,5])
  
Y_pred = pd.Series([0.6,1.29,1.99,2.69,3.4])

MSE(Y_true, Y_pred)
score(Y_true, Y_pred)"""
