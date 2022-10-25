from ConvNN import ConvNN
from Utility import *
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Verificando si torch.cuda está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


#Carga de datos
names, x_train, y_train, x_test, y_test = load_data()

y_train = y_train.type(torch.LongTensor)
y_test = y_test.type(torch.LongTensor)

loss_train = []
loss_test = []

#Creando red neuronal
convNN = ConvNN()
lossFunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(convNN.parameters(),0.001)

# Entrenamiento del modelo
for epoch in range(0, 250):
    # Training
        
    optimizer.zero_grad()
    pred = convNN(x_train)
    print(pred.shape)
    print(y_train.shape)
    loss = lossFunction(pred, y_train.squeeze())
    loss.backward()
    optimizer.step()
    print(loss)

    # Validation
    with torch.no_grad():
       loss2 = lossFunction( convNN(x_test), y_test.squeeze())
       
    loss_train.append(loss.item())
    loss_test.append(loss2.item())
