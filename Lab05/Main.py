from ConvNN import ConvNN
from Utility import *
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import classification_report

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

entrenado = False

if entrenado == False:
    
    convNN.train()
    # Entrenamiento del modelo
    for epoch in range(0, 200):
        
        x_train.to(device)
        y_train.to(device)
        
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
            
           x_test.to(device)
           y_test.to(device) 
            
           loss2 = lossFunction( convNN(x_test), y_test.squeeze())
           
        loss_train.append(loss.item())
        loss_test.append(loss2.item())
        
        if epoch % 50 == 0:
               torch.save(convNN.state_dict(), f'convNNEpoca{epoch}.pt')

else:
    convNN.load_state_dict(torch.load('Model_final.pt'))
    


y_true = []
y_pred = []

y_pred_out = convNN(x_test)

y_pred_out = (torch.max(torch.exp(y_pred_out), 1)[1]).data.cpu().numpy()
y_pred.extend(y_pred_out) # Save Prediction
    
y_test = y_test.data.cpu().numpy()
y_true.extend(y_test) # Save Truth
    

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in names],
                     columns = [i for i in names])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')

print(classification_report(y_true, y_pred, target_names=names))