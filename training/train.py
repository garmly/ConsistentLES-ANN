import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from compute_tau import compute_tau
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
        self.layer1 = nn.Linear(2, 20)
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

# Number of snapshots
num_snapshots = 10

# Training loop
for snapshot in range(num_snapshots):
    # Load the data
    snapshot = snapshot + 1
    tau = read_SGS(f"./out/filtered/SGS/Tau/t{snapshot}.csv", 16, 16, 16)
    R = read_SGS(f"./out/filtered/SGS/R/t{snapshot}.csv", 16, 16, 16)
    S = read_SGS(f"./out/filtered/SGS/S/t{snapshot}.csv", 16, 16, 16)
    delta = read_SGS(f"./out/filtered/delta/t{snapshot}.csv", 16, 16, 16)

    R = torch.tensor(R, dtype=torch.float32)
    S = torch.tensor(S, dtype=torch.float32)

    for epoch in range(num_epochs):
        # Zero the gradients (reset the gradients for each batch)
        optimizer.zero_grad()

        # Forward pass
        input = torch.cat((R, S), dim=1)  # Concatenate R, S, delta along the second dimension
        pred = model(input)  # Compute the output of the model

        R = R.detach().numpy()
        S = S.detach().numpy()
        pred = pred.detach().numpy()
        tau = compute_tau(R, S, 2 * np.pi / R.shape[0], pred)  # Compute the SGS stress tensor
        tau_del = tau + delta  # Add delta to the SGS stress tensor
        tau_del = torch.tensor(tau_del, dtype=torch.float32)  # Convert tau_del to a tensor

        # Compute the loss
        loss = loss_function(pred, tau_del)

        # Backpropagation
        loss.backward()

        # Update the model's parameters
        optimizer.step()

        R = torch.tensor(R, dtype=torch.float32)
        S = torch.tensor(S, dtype=torch.float32)

        # Print the loss for monitoring training progress
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    R = R.detach().numpy()
    S = S.detach().numpy()