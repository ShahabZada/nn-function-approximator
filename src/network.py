import torch
import torch.nn as nn

# Define the neural network class
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),  
            nn.ReLU(),
            nn.Linear(32, 32),  
            nn.ReLU(),
            nn.Linear(32, 32),  
            nn.ReLU(),
            nn.Linear(32, 32),  
            nn.ReLU(),
            nn.Linear(32, 1)   
        )
        
    def forward(self, x):
        return self.layers(x)
    

class RegressionNet2D(nn.Module):
    def __init__(self):
        super(RegressionNet2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32), 
            nn.ReLU(),
            nn.Linear(32, 32),  
            nn.ReLU(),
            nn.Linear(32, 32),  
            nn.ReLU(),
            nn.Linear(32, 32),  
            nn.ReLU(),
            nn.Linear(32, 1)    
        )
        
    def forward(self, x):
        return self.layers(x)

