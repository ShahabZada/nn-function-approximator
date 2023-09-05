import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import save_file
from fourier_network import RegressionNet2D
# import cv2
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from dataset import NNApproxDatasetFourier
from torch.optim import lr_scheduler

n_fourier_features = 16

device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")

dataset = NNApproxDatasetFourier(n_fourier_features)

train_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

for X,y in train_dataloader:
    # print(torch.sum(y))
    X= X.to(device)
    y= y.to(device)

# 
# print(X.shape)
# exit()

model = RegressionNet2D(n_fourier_features=n_fourier_features).to(device)




# Define your initial learning rate
initial_lr = 0.02

# Create your optimizer
# optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# Create a learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.85)

# Define the loss function (mean squared error) and the optimizer (e.g., stochastic gradient descent)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# criterion =nn.SmoothL1Loss(0.1)





# Training loop
num_epochs = 20000

y_pred_list = []

for epoch in range(num_epochs):
    # Forward pass
    # for X, y in train_dataloader:
    # X, y = X.to(device), y.to(device)
    y_pred = model(X)
    
    # Compute the loss
    loss = criterion(y_pred, y)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch<2000:
        y_pred_np = y_pred.detach().cpu().numpy()
        y_pred_list.append(y_pred_np.reshape(dataset.height, dataset.width))
    else:
        if epoch%100==0:
            y_pred_np = y_pred.detach().cpu().numpy()
            y_pred_list.append(y_pred_np.reshape(dataset.height, dataset.width))

    # if (epoch + 1) % 1000 == 0:
    scheduler.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    
    # if loss.item()<0.015:
    #     break
        



save_file('cache_data/cache_arrays.pkl',y_pred_list)
# Test the model


batch_size = len(dataset)

test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Get the entire dataset in a single batch
for batch_input, batch_output in test_dataloader:
    
    batch_input = batch_input.to(device)

    predicted_output = model(batch_input)



y_pred_np = predicted_output.detach().cpu().numpy()

print(dataset.height, dataset.width)
plt.imshow(y_pred_np.reshape(dataset.height, dataset.width), cmap='CMRmap')
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()



