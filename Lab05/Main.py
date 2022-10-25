from ConvNN import ConvNN
from Utility import *
import torch
from torch import nn
import matplotlib.pyplot as plt

#Verificando si torch.cuda está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


#Carga de datos
names, x_train, y_train, x_test, y_test = load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Creando red neuronal
convNN = ConvNN()

# Historial de entrenamiento
loss_hist = []
train_acc_hist = []
val_acc_hist = []

VAL_SIZE = 1000
TRAIN_SIZE = 1000

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(convNN.parameters(),0.001)

# Entrenamiento del modelo
for epoch in range(0, 250):
    # Training
    optimizer.zero_grad()
    
    pred = convNN(x_train)
    
    print(pred.shape)
    print("Holis1")
    loss = lossFunction(pred, y_train.unsqueeze(1))
    print("Holis2")
    loss.backward()
    optimizer.step()

    
    train_correct = (torch.argmax(pred, dim=1) == torch.argmax(
        y_train, 1)).type(torch.float).sum().item()

    # Validation
    with torch.no_grad():
        pred = convNN(x_test)

        val_correct = (torch.argmax(pred, dim=1) == torch.argmax(
            y_test, 1)).type(torch.float).sum().item()

        train_acc_hist.append(train_correct / TRAIN_SIZE)
        val_acc_hist.append(val_correct / VAL_SIZE)

    # Report

    print(f'''
    
    Epoch #{epoch}
    Loss                {loss}
    Train Correct:      {train_correct}
    Train Acc:          {train_acc_hist[-1]}
    Val Correct         {val_correct}
    Val Acc:            {val_acc_hist[-1]}
    ''')


plt.plot(range(250), val_acc_hist, label="Validation")
plt.plot(range(250), train_acc_hist,  label="Training")
plt.title('Accuracy')
plt.legend()
plt.show()