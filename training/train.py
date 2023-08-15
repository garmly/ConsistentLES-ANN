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
        self.layer1 = nn.Linear(8, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 20)
        self.layer4 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
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
num_batches = 3

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_num in range(num_batches):
        # Load the data
        tau = read_SGS(f"./out/filtered/SGS/Tau/t{batch_num + 1}.csv", 64, 64, 64)
        R = read_SGS(f"./out/filtered/SGS/R/t{batch_num + 1}.csv", 64, 64, 64)
        S = read_SGS(f"./out/filtered/SGS/S/t{batch_num + 1}.csv", 64, 64, 64)
        delta = read_SGS(f"./out/filtered/delta/t{batch_num + 1}.csv", 64, 64, 64)
        
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                for k in range(R.shape[2]):
                    # Convert the data to tensors
                    R_tensor = torch.tensor(R[i,j,k], dtype=torch.float32, requires_grad=True)
                    S_tensor = torch.tensor(S[i,j,k], dtype=torch.float32, requires_grad=True)
                    tau_tensor = torch.tensor(tau[i,j,k], dtype=torch.float32, requires_grad=True)

                    # Zero the gradients (reset the gradients for each batch)
                    optimizer.zero_grad()

                    # get 6 coefficients for model
                    input = torch.tensor([torch.trace(S_tensor),
                                          torch.trace(R_tensor**2),
                                          torch.trace(S_tensor**3),
                                          torch.trace(torch.mul(S_tensor,R_tensor**2)),
                                          torch.trace(torch.mul(S_tensor**2,R_tensor**2)),
                                          torch.trace(torch.mul(S_tensor**3,R_tensor**2)),
                                          1e-6,
                                          3**0.5 * np.pi * 2 / 64], 
                                         dtype=torch.float32)
                    
                    # Forward pass
                    nu_t = model(input)[0]  # Compute the output of the model
                    pred = tau_6c_funct.apply(nu_t, S_tensor)  # Compute the predicted SGS stress tensor
                    tau_del = tau_tensor #+ delta  # Add delta to the SGS stress tensor

                    # Compute the loss
                    loss = loss_function(pred, tau_del)
                    total_loss += loss.item()

                    # Backpropagation
                    loss.backward()

                    # Update the model's parameters
                    optimizer.step()

                    #for name, param in model.named_parameters():
                    #    if param.requires_grad:
                    #        print(name, torch.max(param.grad))

    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / num_batches / (64**3):.4f}")