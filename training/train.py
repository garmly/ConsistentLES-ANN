import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture
class SGS_ANN(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear(3, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 20)
        self.layer4 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.output_layer(x)
        return x

# Initialize the neural network
model = SGS_ANN()

# Define the loss function (cost function)
loss_function = nn.MSELoss()

# Define the optimizer as stochastic gradient descent (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Number of training epochs
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Zero the gradients (reset the gradients for each batch)
    optimizer.zero_grad()

    # Forward pass
    input_data = torch.cat((R, S, D), dim=1)  # Concatenate R, S, D along the second dimension
    output_data = model(input_data)

    # Compute the loss
    loss = loss_function(output_data, T)

    # Backpropagation
    loss.backward()

    # Update the model's parameters
    optimizer.step()

    # Print the loss for monitoring training progress
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
