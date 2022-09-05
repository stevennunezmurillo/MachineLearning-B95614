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
matrix_V = result.matrix_V()
#inercias = result.inercias()
"""
plt.scatter(np.ravel(matrix_C[:, 0]), np.ravel(matrix_C[:, 1]), c=[
            'b' if i == 1 else 'r' for i in titanic_data["Survived"]])
plt.xlabel('PCA 1 (%.2f%% inertia)' % (inercias[0],))
plt.ylabel('PCA 2 (%.2f%% inertia)' % (inercias[0],))
plt.title('PCA')
plt.show()
"""

plt.figure(figsize=(15, 15))
plt.axhline(0, color='b')
plt.axvline(0, color='b')
for i in range(0, titanic_data.shape[1]):
    plt.arrow(0, 0, matrix_V[0][i, 0],  # x - PC1
              matrix_V[0][i, 1],  # y - PC2
              head_width=0.05, head_length=0.05)
    plt.text(matrix_V[0][i, 0] + 0.05, matrix_V[0]
             [i, 1] + 0.05, titanic_data.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an), color="b")  # Circle
plt.axis('equal')
plt.title('Correlation Circle')
plt.show()
print("----------------------------------")

scaler = StandardScaler()
df_scaled = scaler.fit_transform(titanic_data)
pca = PCA()
C = pca.fit_transform(df_scaled)
inertia = pca.explained_variance_ratio_
V = pca.transform(np.identity(df_scaled.shape[1]))
"""
plt.scatter(np.ravel(C[:, 0]), np.ravel(C[:, 1]), c=[
            'b' if i == 1 else 'r' for i in titanic_data["Survived"]])
plt.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
plt.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[0],))
plt.title('PCA')
plt.show()
"""
"""
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
plt.title('Correlation Circle')
plt.show()
"""
