import numpy as np


class myPCA:

    def __init__(self, matrix):
        self.matrix_data = matrix

    def centrar_reducir(self):
        
        promedio = self.matrix_data.mean(axis=0)
        desviacion = self.matrix_data.std(axis=0)

        for columna in range(len(self.matrix_data[0])):
            for fila in range(len(self.matrix_data)):
                self.matrix_data[fila][columna] = (
                    self.matrix_data[fila][columna]-promedio[columna])/desviacion[columna]

    def matrix_correlaciones(self):
        
        transpuesta = self.matrix_data.transpose()
        matrix_correlacion = 1/len(self.matrix_data) * \
            np.dot(transpuesta, self.matrix_data)
        return matrix_correlacion

    def matrix_V(self):
        
        valores_propios, vectores_propios = np.linalg.eig(
            self.matrix_correlaciones())

        idx = valores_propios.argsort()[::-1]
        valores_propios = valores_propios[idx]
        vectores_propios = vectores_propios[:, idx]
        return [vectores_propios, valores_propios]

    def matrix_C(self):

        matrix_C = np.dot(self.matrix_data, self.matrix_V()[0])
        return matrix_C

    def inercias(self):
        
        valores_propios = self.matrix_V()[1]

        for i in range(len(valores_propios)):
            valores_propios[i] = valores_propios[i]/len(self.matrix_data[0])

        return valores_propios
