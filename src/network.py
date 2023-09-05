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
            nn.Linear(2, 4), 
            nn.LeakyReLU(0.1),
            nn.Linear(4, 16),  
            nn.LeakyReLU(0.1),
            nn.Linear(16, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 16),  
            nn.LeakyReLU(0.1),
            nn.Linear(16, 4),  
            nn.LeakyReLU(0.1),
            nn.Linear(4, 2),  
            nn.LeakyReLU(0.1),
            nn.Linear(2, 1),
            # nn.Sigmoid()    
        )
        
    def forward(self, x):
        return self.layers(x)

