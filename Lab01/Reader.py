import pandas as pd

fileName = 'C:/Users/B95614/Desktop/ML/MachineLearning-B95614/Lab01/titanic.csv'
titanicData = pd.read_csv(fileName, header = 0)


titanicData = titanicData.drop(['Ticket', 'Name', 'PassengerId', 'Fare', 'Embarked', 'Cabin'], axis = 1)

titanicData = titanicData.dropna()

titanicData = pd.get_dummies(titanicData, columns=['Sex'])

matrixTitanic = titanicData.to_numpy()


print("------------------------------------")
print(titanicData)

print('\nNumpy Array\n----------\n', matrixTitanic)