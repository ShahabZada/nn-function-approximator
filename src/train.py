import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from network import RegressionNet

# Create an instance of the regression model
model = RegressionNet()

# Define the loss function (mean squared error) and the optimizer (e.g., stochastic gradient descent)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# Generate synthetic data for demonstration
# torch.manual_seed(42)
x_data = torch.linspace(-6, 6, steps=100)
x_data = torch.reshape(x_data, (x_data.shape[0], 1))
# print(torch.randn(100, 1).shape, x_data.shape)
y_true = torch.sin(x_data)#2 * x_data + 1 #+ 0.1 * torch.randn(100, 1)  # y = 2x + 1 with noise

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x_data)
    
    # Compute the loss
    loss = criterion(y_pred, y_true)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
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

# Test the model
# test_input = torch.tensor([[3.0]])  # Test with input x = 3.0
x_data_test = torch.linspace(-6, 6, steps=1000)
x_data_test = torch.reshape(x_data_test, (x_data_test.shape[0], 1))
predicted_output = model(x_data_test)
# print(f'Predicted output for x = 3.0: {predicted_output.item():.4f}')



# Convert tensors to numpy arrays for plotting
x_data_np = x_data.numpy()
x_data_test_np = x_data_test.numpy()

y_true_np = y_true.numpy()
y_pred_np = predicted_output.detach().numpy()



# Scatter plot
plt.scatter(x_data_np, y_true_np,s=2, color='blue', label='True Data')
plt.scatter(x_data_test_np, y_pred_np,s=2, color='orange', label='Predicted Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot  Data')
plt.legend()
plt.show()