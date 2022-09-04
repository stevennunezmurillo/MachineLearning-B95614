import pandas as pd
from myPCA import myPCA
from sklearn.preprocessing import StandardScaler


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

print("------------------------------------")
"""
scaler = StandardScaler()
df_scaled = scaler.fit_transform(titanic_data)
print(df_scaled)
"""
result.matrix_V()
