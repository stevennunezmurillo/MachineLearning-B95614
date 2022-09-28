from msilib.schema import ServiceInstall
from ssl import _create_default_https_context
from tkinter import Y
from xml.sax import SAXNotSupportedException
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd


#Método de Mínimos cuadrados, error cuadrático medio.
def MSE(y_true, y_predict):
    
    error = 0

    for index in range(len(y_true)):
        error += ((y_predict[index] - y_true[index])**2)
    
    error *= 1/ len(y_true)
    
    return error
    

#Método de Coeficiente de determinación (R2) 
def score(y_true, y_predict):
    
    #r2 = r2_score(y_true, y_predict)

    
    first_sum = sum(((y_true[0] - y_predict[0])**2))
    mean = y_true[0].mean()
    second_sum = sum(((y_true[0] - mean)**2))
    
    score_r2 = (second_sum-first_sum)/second_sum
    
    return r2_score(y_true[0], y_predict[0])
     
        

