import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from compute_tau import *
from read_SGS import read_SGS

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device for training.")

# Define the neural network architecture
class SGS_ANN(nn.Module):
    def __init__(self):
        super(SGS_ANN, self).__init__()
        self.requires_grad_(True)
        self.layer1 = nn.Linear(16*16*16*3*3*2, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.layer4 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 3)

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
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Number of training epochs
num_epochs = 10

# Number of snapshots
num_batches = 100

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    # change to 6-1 model
    # make pointwise
    # make draft presentation for group meeting

    for batch_num in range(num_batches):
        # Load the data
        tau = read_SGS(f"./out/filtered/SGS/Tau/t{batch_num + 1}.csv", 16, 16, 16)
        R = read_SGS(f"./out/filtered/SGS/R/t{batch_num + 1}.csv", 16, 16, 16)
        S = read_SGS(f"./out/filtered/SGS/S/t{batch_num + 1}.csv", 16, 16, 16)
        delta = read_SGS(f"./out/filtered/delta/t{batch_num + 1}.csv", 16, 16, 16)
        
        R_tensor = torch.tensor(R, dtype=torch.float32, requires_grad=True)
        S_tensor = torch.tensor(S, dtype=torch.float32, requires_grad=True)
        tau = torch.tensor(tau, dtype=torch.float32, requires_grad=True)

        # Zero the gradients (reset the gradients for each batch)
        optimizer.zero_grad()

        # Forward pass
        input = torch.cat((R_tensor[:, :, :, :, :, None], S_tensor[:, :, :, :, :, None]), dim=-1)  # Concatenate R, S, delta along the last dimension
        input = torch.flatten(input)  # Flatten the input
        pred = model(input)  # Compute the output of the model
        pred = tau_funct.apply(pred, R_tensor, S_tensor, np.pi*2 / 16)  # Compute the predicted SGS stress tensor
        tau_del = pred #+ delta  # Add delta to the SGS stress tensor

        # Compute the loss
        loss = loss_function(pred, tau)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()

        # Update the model's parameters
        optimizer.step()

        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.grad)

    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / num_batches:.4f}")