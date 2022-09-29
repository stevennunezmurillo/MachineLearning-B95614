"""
Created on Tue Sep 27 22:26:26 2022

@author: Steven Nuñez -B95614
"""

from xml.sax import SAXNotSupportedException

import utility 
import random
import numpy as np
import pandas as pd

class LinearRegression:
    
        
    def fit(self, x, y, max_epochs=100, threshold=0.01, learning_rate=0.001, momentum=0, decay=0, error='mse', regularization='none', lambdas=0):
        
        #Se pasa de dataframe a matriz numpy
        x_matrix = x.to_numpy()
        
        #Creación de coeficientes aleatorios para la primer iteración
        coeficients = np.array([random.sample(range(len(x_matrix[0])),len(x_matrix[0]))])
        coeficients = np.transpose(coeficients)

        #Variables locales
        epoch = 0
        epoch_error = []
        list_changes = []
        
        #----------------------------------------------------------------------------------
        #Algoritmo de RG
        while epoch < max_epochs:

            #Calculo de las predicciones
            xc_pred = self.xc_matrix_gen(x_matrix, coeficients)
            
            #Se selecciona las mejoras para el calculo del error
            if (regularization == 'l1' or regularization == 'lasso'):
                current_error = self.lasso(utility.MSE(y, xc_pred), lambdas, coeficients)
            else:
                if (regularization == 'l2' or regularization == 'ridge'):
                    current_error = self.ridge(utility.MSE(y, xc_pred), lambdas, coeficients)
                else:
                    current_error = utility.MSE(y, xc_pred)
                    
            #Se almacena el error para tener control del error en cada epoca
            epoch_error.append(current_error)
            #print(epoch_error[epoch])
            #Se calcula el cambio, con la derivada
            change = self.derivatives_errorc(xc_pred, y, x_matrix)
            
            #Se almacena para tener controlo de los cambios en las diferentes epocas
            list_changes.append(change)
            
            #Si no es la primer epoca, ver si el cambio en el error entre una época y la anterior sea mayor a un umbral
            # y se calcula los coeficientes nuevos con el momentum o si es la primer epoca se calcula de manera normal.
            if epoch != 0:

                if (abs(epoch_error[epoch]-epoch_error[epoch-1]) < threshold):
                    return coeficients
                
                coeficients = self.momentum_coeficients(coeficients, learning_rate, change, momentum, list_changes[epoch-1])
            else:
                coeficients = self.new_coeficients(coeficients, change, learning_rate)

            #Manejo del decay
            learning_rate = learning_rate/(1+decay)
            
            #Siguiente epoca
            epoch+=1

        return coeficients
    
    #----------------------------------------------------------------------------------
    #Métodos para el algoritmo de RG
    
    #Calcula el predict de xc
    def xc_matrix_gen(self, x, c):
        return np.dot(x,c)
    
    #Se calcula el cambio con la derivada respecto a los coeficientes
    def derivatives_errorc(self, xc_predict, y_true, x_ones):
        return (2/len(xc_predict))*np.dot((xc_predict-y_true).transpose(),x_ones).transpose()

    #Se calcula los nuevos coeficientes
    def new_coeficients(self, coeficients, error_rg, eta):
        return coeficients - (error_rg *eta)
    
    #Calculo de los coeficientes con momentum
    def momentum_coeficients(self, coeficients, eta, change, momentum, last_change):
        return coeficients - eta * (change + (momentum *last_change))
        
    #Mejora para el calculo del error con l1
    def lasso(self, error_mse, lambdas, coeficients):
        return error_mse + lambdas + sum(abs(coeficients.transpose()[0]))

    #Mejora para el calculo del error con l2
    def ridge(self, error_mse, lambdas, coeficients):
        return error_mse + lambdas + sum((coeficients.transpose()[0])**2)

    #----------------------------------------------------------------------------------
    #Método para el calculo de predicciones
    def predict(self, x, coeficients):
        
        x_matrix = x.to_numpy()
        return np.dot(x_matrix, coeficients)
        



