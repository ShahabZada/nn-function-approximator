import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import save_file
from network import RegressionNet2D
# import cv2
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from dataset import NNApproxDataset


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

dataset = NNApproxDataset()

train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


model = RegressionNet2D().to(device)


# Define the loss function (mean squared error) and the optimizer (e.g., stochastic gradient descent)
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# Training loop
num_epochs = 100

y_pred_list = []

for epoch in range(num_epochs):
    # Forward pass
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        
        # Compute the loss
        loss = criterion(y_pred, y)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # y_pred_np = y_pred.detach().numpy()
        # y_pred_list.append(y_pred_np)


    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        



# save_file('cache_data/cache_arrays.pkl',(x_data_np,y_true_np,y_pred_list))
# Test the model


batch_size = len(dataset)

test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Get the entire dataset in a single batch
for batch_input, batch_output in test_dataloader:
    
    batch_input = batch_input.to(device)

    predicted_output = model(batch_input)



y_pred_np = predicted_output.detach().numpy()

print(dataset.height, dataset.width)
plt.imshow(y_pred_np.reshape(dataset.height, dataset.width), cmap='CMRmap')
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()



