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
        valores_propios, vectores_propios = np.linalg.eigh(
            self.matrix_correlaciones())
        print(valores_propios)
        print(vectores_propios)
        pass

    def matrix_C(self):
        pass
