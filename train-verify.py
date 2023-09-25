import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from training.compute_tau import *
from training.compute_closure import *
from training.read_SGS import read_SGS

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device for training.")
path = './in'

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
        x = torch.relu(self.layer1(x))
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
num_epochs = 200

loss_list = []
batch_num = 0

# Read data from .csv files
tau = read_SGS(f"{path}/filtered/SGS/Tau/t{batch_num + 1}.csv", 64, 64, 64)
R = read_SGS(f"{path}/filtered/SGS/R/t{batch_num + 1}.csv", 64, 64, 64)
S = read_SGS(f"{path}/filtered/SGS/S/t{batch_num + 1}.csv", 64, 64, 64)
delta = read_SGS(f"{path}/filtered/delta/t{batch_num + 1}.csv", 64, 64, 64)

# Convert the data to tensors
R_tensor = torch.tensor(R[31,31,31], dtype=torch.float32, requires_grad=True)
S_tensor = torch.tensor(S[31,31,31], dtype=torch.float32, requires_grad=True)
tau_tensor = torch.tensor(tau[31,31,31], dtype=torch.float32, requires_grad=True)
delta_tensor = torch.tensor(delta[31,31,31], dtype=torch.float32, requires_grad=True)

epoch = 0
total_loss = 0

# Training loop
for i in range(num_epochs):
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
    output = model(input)  # Compute the output of the model

    # Compute the predicted SGS stress tensor
    pred_nu = tau_nu_funct.apply(output, S_tensor)
    pred = nu_deriv_funct.apply(output,tau,delta_tensor,[31,31,31],2*np.pi/64)

    # Compute the loss
    loss = loss_function(pred_nu, dtau_del(tau,delta_tensor,[31,31,31],2*np.pi/64))
    loss_list.append(loss.item())
    total_loss += loss.item()

    print(f'\rEpoch: {epoch}/{num_epochs} | Epoch Progress: {epoch/num_epochs*100:5.2f}% | Loss: {loss.item():10.4f}', end='')

    # Backpropagation
    loss.backward()
    optimizer.step()

    epoch += 1

# Print predicted and actual closure terms
print(f'\nPredicted closure term:\n{pred}')
print(f'Actual closure term:\n{dtau_del(tau,delta_tensor,[31,31,31],2*np.pi/64)}')

# Print predicted and actual SGS stress tensors
print(f'\nPredicted SGS stress tensor:\n{pred_nu*S_tensor}')
print(f'Actual SGS stress tensor:\n{tau_tensor}')
print(f'Actual S tensor:\n{S_tensor}')

conv_size = 1
plt.plot(np.convolve(loss_list, np.ones(conv_size), 'valid') / conv_size, label='Training')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
plt.close()