import torch
from torch import nn

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        
        
        
        self.flatten = nn.Flatten()
        self.process = nn.Sequential(
            
            nn.Conv2d(3, 5, 3, padding="same"),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(5, 10, 3, padding="same"),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, 3, padding="same"),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        
        self.denseProcess = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Linear(160, 10),
            nn.ReLU(),
            nn.Softmax(),
        )
            
       
        
        
    def forward(self, x):
        
        x = self.process(x)
        x = self.flatten(x)
        x = x.view(x.shape, 1)
        print("Tensooor")
        print(x.shape)
        out =  self.denseProcess(x)
        
        
        return out
