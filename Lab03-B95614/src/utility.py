"""
Created on Tue Sep 27 22:33:25 2022

@author: Steven Nuñez -B95614
"""

from msilib.schema import ServiceInstall
from ssl import _create_default_https_context
from tkinter import Y
from xml.sax import SAXNotSupportedException

import numpy as np
import pandas as pd


#Método de Mínimos cuadrados, error cuadrático medio.
def MSE(y_true, y_predict):
    
    error = 0

    for index in range(len(y_true)):
        error += ((y_predict[index] - y_true[index])**2)
    
    error *= 1/ len(y_true)
    
    return error
    

def score(y_true, y_predict):
    
    
    first_sum = sum(((y_true[0] - y_predict[0])**2))
    mean = y_true[0].mean()
    second_sum = sum(((y_true[0] - mean)**2))
    
    score_r2 = (second_sum-first_sum)/second_sum
    
    return score_r2
     
        

