"""
Created on Tue Sep 27 22:33:25 2022

@author: Steven Nuñez -B95614
"""
from xml.sax import SAXNotSupportedException
from sklearn.model_selection import train_test_split
from linearRegression import LinearRegression 

import numpy as np
import pandas as pd
import utility 

def main():

    x = pd.read_csv("../data/fish_perch.csv")
    y = np.array([x['Weight']]).transpose()
    
    x.drop(['Weight'], axis=1, inplace = True)
    
    linear_regression = LinearRegression()
    
    #Usando el splir de sklearn
    #x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 21)
    
        #Otra de las semillas usadas
    #x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 77)
    
        #Otra de las semillas usadas
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 14)
    

    #Se agrega el bias  
    x_train.insert(0, "ones", 1, allow_duplicates=False)  
    x_test.insert(0, "ones", 1, allow_duplicates=False)
    
    #print(x_test)
    #print(y_test)
    
    #----------------------------------------------------------------------------------
    #Inicio de test con la misma semilla
    
    print("\n.----First test----.")
    coeficients = linear_regression.fit(x_train, y_train, max_epochs=1e10, learning_rate=1e-4, momentum=0.7, decay=1e-5, error='mse', regularization='l1', lambdas=0.7)
    prediction = linear_regression.predict(x_test, coeficients)
    print(utility.score(y_test.transpose(), prediction.transpose()))
    print(".---------------------.\n")
    
    
    print(".----Second test----.")
    coeficients = linear_regression.fit(x_train, y_train, max_epochs=1e9, threshold=0.01, learning_rate=1e-4, momentum=0.9, decay=1e-10, error='mse', regularization='none', lambdas=0.2)
    prediction = linear_regression.predict(x_test, coeficients)
    print(utility.score(y_test.transpose(), prediction.transpose()))
    print(".---------------------.\n")
    
    print(".----Third test----.")
    coeficients = linear_regression.fit(x_train, y_train, max_epochs=1e5, threshold=0.01, learning_rate=1e-4, momentum=0.9, decay=1e-9, error='mse', regularization='l2', lambdas=0)
    prediction = linear_regression.predict(x_test, coeficients)
    print(utility.score(y_test.transpose(), prediction.transpose()))
    print(".---------------------.\n")
    
    print(".----Fourt test----.")
    coeficients = linear_regression.fit(x_train, y_train, max_epochs=1e5, threshold=0.01, learning_rate=1e-4, momentum=0.9, decay=1e-9, error='mse', regularization='l1', lambdas=0)
    prediction = linear_regression.predict(x_test, coeficients)
    print(utility.score(y_test.transpose(), prediction.transpose()))
    print(".---------------------.")

    #Fin de test
    #----------------------------------------------------------------------------------
    
    
    """c.¿Cuál fue la combinación de parámetros que le proveyó el mejor resultado?
        
        La mejor combinación está dada por el cuarto test con los parametros:
        max_epochs=1e5, threshold=0.01, learning_rate=1e-4, momentum=0.9, decay=1e-9, error='mse', regularization='l1' y lambdas=0
    
        
        El resultado del R2 fue 0.9023977319560136
        
    """
    
    """
       d.¿Qué pasa si utiliza esa misma combinación pero cambia la semilla del train_test_split? 

        Nota: Los split con otras semillas están arribas comentados, para probar descomentar uno y comentar
        el actual
        
        R/Se nota una mejora en algunos casos y en otras más bien una desmejora, es variado.
        
        Por ejemplo se obtuvo un peor resultado en general del R2 con la semilla 14 y los parametros:
        max_epochs=1e5, threshold=0.01, learning_rate=1e-4, momentum=0.9, decay=1e-9, error='mse', regularization='l2', lambdas=0
        el valor de este R2 fue: 0.6517327354025293
        
        
        Por el contrario también se obtuvo el mejor resultado hasta el momento con la semilla 14 y los parametros:
        max_epochs=1e5, threshold=0.01, learning_rate=1e-4, momentum=0.9, decay=1e-9, error='mse', regularization='l1', lambdas=0
        el valor del R2 fue: 0.921773118466989
    """
    
    """
       e.¿Por qué cree que pasa esto?
       
       Que esto suceda es mera aletoriedad ya que implantar una semilla solo permite que los conjuntos generados sean diferentes de una
       semilla a otra, entonces una puede entregar mejores set de entrenamiento haciendo que se obtengan coeficientes más certeros del fit
       que arrojen a una mejor predicción, sin embargo, no se nota un patrón en especifico que pueda revelar algo más que la simple aletoriedad
       
    """
if __name__ == "__main__":
    main()