import torch
import torch.nn as nn


class RegressionNet2D(nn.Module):
    def __init__(self, n_fourier_features):
        super(RegressionNet2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_fourier_features, 32), 
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32),  
            nn.LeakyReLU(0.1),
            nn.Linear(32, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 32),  
            nn.LeakyReLU(0.1),
            nn.Linear(32, 8),  
            nn.LeakyReLU(0.1),
            nn.Linear(8, 4),  
            nn.LeakyReLU(0.1),
            nn.Linear(4, 1),
            # nn.Sigmoid()    
        )
        
    def forward(self, x):
        return self.layers(x)

