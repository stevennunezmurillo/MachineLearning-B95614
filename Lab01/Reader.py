import pandas as pd
from myPCA import myPCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np


file_name = 'C:/Users/snmtr/Videos/UCR/ML/MachineLearning-B95614/Lab01/titanic.csv'

titanic_data = pd.read_csv(file_name, header=0)

titanic_data = titanic_data.drop(
    ['Ticket', 'Name', 'PassengerId', 'Fare', 'Embarked', 'Cabin'], axis=1)

titanic_data = titanic_data.dropna()

titanic_data = pd.get_dummies(titanic_data, columns=['Sex'])

matrix_titanic = titanic_data.to_numpy()


# print(titanic_data)

#print('\nNumpy Array\n----------\n', matrix_titanic)


result = myPCA(matrix_titanic)
result.centrar_reducir()

matrix_C = result.matrix_C()
#inercias = result.inercias()
"""
plt.scatter(np.ravel(matrix_C[:, 0]), np.ravel(matrix_C[:, 1]), c=[
            'b' if i == 1 else 'r' for i in titanic_data["Survived"]])
plt.xlabel('PCA 1 (%.2f%% inertia)' % (inercias[0],))
plt.ylabel('PCA 2 (%.2f%% inertia)' % (inercias[0],))
plt.title('PCA')
plt.show()
"""
print("----------------------------------")

scaler = StandardScaler()
df_scaled = scaler.fit_transform(titanic_data)
pca = PCA()
C = pca.fit_transform(df_scaled)
inertia = pca.explained_variance_ratio_
V = pca.transform(np.identity(df_scaled.shape[1]))
plt.scatter(np.ravel(C[:, 0]), np.ravel(C[:, 1]), c=[
            'b' if i == 1 else 'r' for i in titanic_data["Survived"]])
plt.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
plt.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[0],))
plt.title('PCA')
plt.show()
