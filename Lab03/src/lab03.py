# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:33:25 2022

@author: Steven Nuñez -B95614
"""
from xml.sax import SAXNotSupportedException
from sklearn.model_selection import train_test_split
from linearRegression import LinearRegression 

import numpy as np
import pandas as pd
import predict 

def main():

    x = pd.read_csv("../data/fish_perch.csv")
    y = np.array([x['Weight']]).transpose()
    
    x.drop(['Weight'], axis=1, inplace = True)
    
    linear_regression = LinearRegression()
    
    #Usando el splir de sklearn
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 21)
    #Se agrega el bias  
    x_train.insert(0, "ones", 1, allow_duplicates=False)  
    x_test.insert(0, "ones", 1, allow_duplicates=False)
    
    #print(x_test)
    #print(y_test)
    
    print(".----First test----.")
    coeficients = linear_regression.fit(x_train, y_train, max_epochs=1e10, learning_rate=1e-4, momentum=0.7, decay=1e-5, error='mse', regularization='l1', lambdas=0.7)
    prediction = linear_regression.predict(x_test, coeficients)
    print(predict.score(y_test.transpose(), prediction.transpose()))
    print(".------------.")

    print(".----Second test----.")
    coeficients = linear_regression.fit(x_train, y_train, max_epochs=1e5, threshold=0.01, learning_rate=1e-5, momentum=0, decay=0, error='mse', regularization='l1', lambdas=0)
    prediction = linear_regression.predict(x_test, coeficients)
    print(predict.score(y_test.transpose(), prediction.transpose()))
    print(".------------.")
    print(".----Third test----.")
    coeficients = linear_regression.fit(x_train, y_train, max_epochs=1e4, threshold=0.01, learning_rate=1e-5, momentum=0, decay=0, error='mse', regularization='.2', lambdas=0)
    prediction = linear_regression.predict(x_test, coeficients)
    print(predict.score(y_test.transpose(), prediction.transpose()))
    print(".------------.")
    print(".----Fourt test----.")
    coeficients = linear_regression.fit(x_train, y_train, max_epochs=1e5, threshold=0.01, learning_rate=1e-8, momentum=0, decay=0, error='mse', regularization='none', lambdas=0)
    prediction = linear_regression.predict(x_test, coeficients)
    print(predict.score(y_test.transpose(), prediction.transpose()))
    print(".------------.")
    
if __name__ == "__main__":
    main()