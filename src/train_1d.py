import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import save_file
from network import RegressionNet

# Create an instance of the regression model
model = RegressionNet()

# Define the loss function (mean squared error) and the optimizer (e.g., stochastic gradient descent)
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Generate synthetic data for demonstration
# torch.manual_seed(42)
# x_data = torch.linspace(-9, 9, steps=1000)
# x_data = torch.reshape(x_data, (x_data.shape[0], 1))
# print(torch.randn(100, 1).shape, x_data.shape)
# y_true = torch.sin(x_data)
# y_true = torch.abs(x_data)


x_data = torch.linspace(0.05, 0.2, steps=1000)
x_data = torch.reshape(x_data, (x_data.shape[0], 1))
y_true = torch.sin(1/x_data)

# Training loop
num_epochs = 5000


x_data_np = x_data.numpy()
y_true_np = y_true.numpy()
y_pred_list = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x_data)
    
    # Compute the loss
    loss = criterion(y_pred, y_true)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    y_pred_np = y_pred.detach().numpy()
    y_pred_list.append(y_pred_np)


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        # x_data_np = x_data.numpy()
        # y_true_np = y_true.numpy()
        # y_pred_np = y_pred.detach().numpy()

        # # Scatter plot
        # plt.scatter(x_data_np, y_true_np, color='blue', label='True Data')
        # plt.scatter(x_data_np, y_pred_np, color='orange', label='Predicted Data')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('Scatter Plot  Data')
        # plt.legend()
        # plt.show()



save_file('cache_data/cache_arrays.pkl',(x_data_np,y_true_np,y_pred_list))
# Test the model



predicted_output = model(x_data)
# print(f'Predicted output for x = 3.0: {predicted_output.item():.4f}')



# Convert tensors to numpy arrays for plotting
x_data_np = x_data.numpy()
y_true_np = y_true.numpy()
y_pred_np = predicted_output.detach().numpy()



# Scatter plot
plt.scatter(x_data_np, y_true_np,s=2, color='blue', label='True Data')
plt.scatter(x_data_np, y_pred_np,s=2, color='orange', label='Predicted Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot  Data')
plt.legend()
plt.show()
