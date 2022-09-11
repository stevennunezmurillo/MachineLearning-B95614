import pandas as pd
from myPCA import myPCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np

#Selección de la ruta del set de datos
file_name = 'C:/Users/snmtr/UCR/ML/MachineLearning-B95614/Lab01/titanic.csv'

#Cargado del set de datos
titanic_data = pd.read_csv(file_name, header=0)

"""
-Se decidió eliminar la columna de Ticket ya que solamente es un identificador del 
    ticket correspondiente al pasajero lo cual no es relevante para decidir si un pasajero
    vive o no, principalmente porque no aporta valor para conocer la "situación" de la persona dentro del 
    barco, es solo un numero.
    
-Respecto al nombre, tampoco se considera algo que pueda influenciar al determinar si la 
    persona vive o no, un nombre es como un identifcador no unico que puede tener la persona
    pero tampoco es un factor de superviencia, incluso en caso de que dos personas tengan el mismo
    nombre y uno se salva, no va implicar que el otro con el mismo nombre también lo va a hacer.
    
-El passengerId al igual que el nombres es un identificador pero que esta vez sí es unico para cada 
    pasajero mas no apunta a ser un factor de supervivencia ya que podría ser un Id aleatorio, o que 
    sea en el orden que entraron al barco, lo cual no tiene relación directa con sobrevivir, más que tampoco
    se podría decir que si entraron primero escogieron mejor posición por ejemplo, ya que las secciones 
    del barco ya estaban seccionadas según el ticket que se comprara (diferentes clases).
    
-El fare es el precio de venta del ticket, podría que este si tenga relación con la supervivencia de las personas
    ya que precios más altos, significaría una "mejor" posición dentro del barco quizás con más salidas de emergencia por ejemplo, sin embargo, se considera que 
    la columna Pclass da una información más fácil de interpretar de donde se ubicaban los pasajeros y en que clase
    que es basicamente lo que haría cambiar el precio del ticket, ver en cual clase se va a estár.
    
-Embarked nos da el puerto en que embarcó el pasajero, esto no se considera de relevancia para el estudio
    ya que los tickets al final de cuentas eran los que indicaban en cual clase viajaba la persona, no nos da
    información de dónde se encontraba siquiera o en qué condiciones se econtraba el pasajero abordo.


-Cabin es un identificador de la cabina del pasajero, no se toma en cuenta a pesar de que en cierta parte brinda información
    de la posible ubicación del pasajero ya que interpretar ese numero es un poco dificl, además se considera que la columna Pclass
    en cierta parte ya nos da esa noción de ubicación del pasajero, porque las cabinas y demás estaban acomodadas por la clase del 
    pasajero.
    
-Se tomaron en cuenta el sexo principalmente porque como hipotesis se puede partir que por cortesía o algo similar
    se pudiera tener cierta prioridad para las mujeres, además al ser una emergencia la capacidad fisica de los hombres y mujeres
    en muchos de los casos varia, la edad se considera importante ya que una persona más joven puede que tenga más capacidad fisica para 
    realizar ciertas acciones que lo puedan poner a salvo o quizás una mentalidad más imprudente, la pclass o clase, esto siempre va a ser un atributo a tomar en cuenta
    ya que tener una clase más alta implica tener más dinero y por consiguiente más accesos y a la vez prioridad, el total de hermanos y esposa abordo y 
    el total de padres e hijos, el tema de familiares es considerado relevante ya que a la hora de las emergencias tiene que pensar no solo en si mismo
    sino en más personas y en algunos casos tiene que limitarse a lo que hagan o por ejemplo que tomar una decición de salvarse o no implique abandonarlos y 
    puede que la persona no opte por esa opción y prefiera morir.

"""


titanic_data = titanic_data.drop(
    ['Ticket', 'Name', 'PassengerId', 'Fare','Embarked', 'Cabin'], axis=1)

#Conversión de las variables categóricas en variables numéricas.
titanic_data = pd.get_dummies(titanic_data, columns=['Pclass','Sex', 'Survived'])

#Eliminación de cualquier entrada que posea datos faltantes
titanic_data = titanic_data.dropna()

#Conviersión de los datos en una matriz de numpy.
matrix_titanic = titanic_data.to_numpy()

"""
Prints del set de datos
"""
#print(titanic_data)
#print('\nNumpy Array\n----------\n', matrix_titanic)

#Instancia de myPCA con la matriz numpy de los datos obtenidos de titanic_data
result = myPCA(matrix_titanic)

#Se aplica sobre el atributo matrix_data de la instancia result el metodo centrar y reducir 
result.centrar_reducir()

#Se obtiene la matriz V a partir de la matriz correlaciones
matrix_V = result.matrix_V()

#Se obtiene la matriz C a partir de la matriz V
matrix_C = result.matrix_C()


inercias = result.inercias()


"""
¿Cuántos grupos de datos parece haber?

    R/ Aparenta haber tres grupos de datos, en la parte izquierda se aglomera un conjunto de puntos unicamente azules
    con representación de un grupo grande de supervivientes, al otro extremo al lado derecho se
    refleja un un grupo de puntos rojos que representan a los no superviventes y en el medio hay un mixto
    entre supervivente y no superviventes.
    
    
¿Qué comportamientos se pueden observar? / ¿Qué podría explicar estos comportamientos? / ¿Qué nos indica el círculo de correlación?

    R/  El circulo de correlación ayuda a interpretar las correlaciones exitentes entre las variables dentro del set de datos y 
    a su vez junto con el scatter plot permite comprender de mejor manera el comportamiento de los grupos. En este caso se puede 
    observar como el grupo de sobrevivientes de mayor tamaño que corresponde al grupo de la izquierda está conformado por mujeres
    mientras que el de la derecha de no sobrevientes son hombres, lo que nos deja pensando en una posible prioridad que se tenía de rescatar a la mujeres
    antes que a los hombres, además de esto gracias al circulo de correlación se aprecia como el grupo mixto que se encuentra en el medio
    en la parte inferior hay un gran grupo de sobrevivientes que pertenecen a las personas de primer clase, mientras que los sobrevivientes 
    de clase media fueron un grupo menor y los de clase baja casi nula, de lo que se puede inferir que la clase social a la cual perteneciera la 
    persona sí fue importante al decidir a quién montar a los botes de emergencia o quizás tenían un acceso más rápido, una apreciación interesante
    es que se observan dos lineas en el scatter plot de personas que no sobrevivieron una de estas corresponde a las personas con esposa y hermanos a bordo  y la 
    otra a si tenía padres e hijos a bordo, por lo tanto se puede interpretar que quizás el contar con familiares a bordo pudo entorpecer las evacuaciones o quizás
    personas con tal de no abandonar a su familia decidía no tomar ciertas oportunidades que los pudo haber salvado. 

¿Qué atributos o características maximizarían mi probabilidad de sobrevivencia?

    R/  Los atributos que parecen sobresalir al tratar de maximizar las probabilidades de 
    sobrevivencia son principalmente:
    
        1) Sex_Famale: El ser mujer.
        2) Pclass_1: Ser de primera clase.
       
"""



#Graficación de los datos sobre los componentes principales coloreada según su valor “Survived_1”
plt.scatter(np.ravel(matrix_C[:, 0]), np.ravel(matrix_C[:, 1]), c=[
            'b' if i == 1 else 'r' for i in titanic_data["Survived_1"]])
plt.xlabel('PCA 1 (%.2f%% inertia)' % (inercias[0]*100,))
plt.ylabel('PCA 2 (%.2f%% inertia)' % (inercias[1]*100,))
plt.title('Mi PCA')
plt.show()

#Graficación del círculo de correlación del modelo.
plt.figure(figsize=(15, 15))
plt.axhline(0, color='b')
plt.axvline(0, color='b')
for i in range(0, titanic_data.shape[1]):
    plt.arrow(0, 0, np.multiply(matrix_V[0][i, 0], matrix_V[1][0]**0.5),  # x - PC1
              np.multiply(matrix_V[0][i, 1], matrix_V[1][1]**0.5),  # y - PC2
              head_width=0.05, head_length=0.05)
    plt.text(np.multiply(matrix_V[0][i, 0], matrix_V[1][0]**0.5) + 0.05, np.multiply(matrix_V[0][i, 1], matrix_V[1][1]**0.5) + 0.05, titanic_data.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an), color="b")  # Circle
plt.axis('equal')
plt.title('Mi círculo correlación')
plt.show()

"""
#------------------------------------------------------------------------------

Implementación con sklearn
"""
#Recolección de datos con sklearn
scaler = StandardScaler()
df_scaled = scaler.fit_transform(titanic_data)
pca = PCA()
C = pca.fit_transform(df_scaled)
inertia = pca.explained_variance_ratio_
V = pca.transform(np.identity(df_scaled.shape[1]))


"""
¿Hay alguna diferencia entre las gráficas? De ser así, ¿por qué cree que ocurrió esto?, ¿Impacta el resultado de alguna manera?

    R/ Sí hay una diferencia, ya que se crea un espejo al comparar los gráficos generados con los producidos por la biblioteca sklearn
    esto sucede ya que la PCA busca máximizar la distancia de representación de datos que al estar en un vector cuenta con varias "dimensiones"
    entonces es como si se tomara una camara y se vieran varios angulos de los datos, al final de cuentas son los mismos datos, los mismos resultados
    pero observados desde otro angulo o perspectiva, los puntos se encuentran distribuidos de la misma manera que es lo que es de mayor interes. Por lo tanto, no 
    impacta en el resultado, ya que son los mismos.

"""
#Graficación de los datos sobre los componentes principales coloreada según su valor “Survived_1”
plt.scatter(np.ravel(C[:, 0]), np.ravel(C[:, 1]), c=[
            'b' if i == 1 else 'r' for i in titanic_data["Survived_1"]])
plt.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0]*100,))
plt.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[1]*100,))
plt.title('Sklearn - PCA')
plt.show()


#Graficación del círculo de correlación del modelo.
plt.figure(figsize=(15, 15))
plt.axhline(0, color='b')
plt.axvline(0, color='b')
for i in range(0, titanic_data.shape[1]):
    plt.arrow(0, 0, V[i, 0],  # x - PC1
              V[i, 1],  # y - PC2
              head_width=0.05, head_length=0.05)
    plt.text(V[i, 0] + 0.05, V[i, 1] + 0.05, titanic_data.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an), color="b")  # Circle
plt.axis('equal')
plt.title('Sklearn - círculo correlación')
plt.show()


