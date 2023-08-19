import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network class
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),  # 1 input neuron, 32 hidden neurons
            nn.ReLU(),
            nn.Linear(32, 64),  # 32 hidden neurons, 64 hidden neurons
            nn.ReLU(),
            nn.Linear(64, 64),  # 32 hidden neurons, 64 hidden neurons
            nn.ReLU(),
            nn.Linear(64, 1)    # 64 hidden neurons, 1 output neuron
        )
        
    def forward(self, x):
        return self.layers(x)

