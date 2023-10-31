import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from training.compute_tau import *
from training.compute_closure import *
from training.read_SGS import *

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
        self.layer1 = nn.Linear(6, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 20)
        self.layer4 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 1)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

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
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Number of training epochs
num_epochs = 200

loss_list = []

# Read data from .csv files
tau = read_SGS_binary(f"{path}/filtered/SGS/Tau/t1.bin")[...,0] /100
R = read_SGS_binary(f"{path}/filtered/SGS/R/t1.bin")[...,0] / 100
S = read_SGS_binary(f"{path}/filtered/SGS/S/t1.bin")[...,0] / 100
delta = read_delta(f"{path}/filtered/delta/t1.bin") / 100

# Convert the data to tensors
R_tensor = torch.tensor(R[21,21,21], dtype=torch.float32, requires_grad=True)
S_tensor = torch.tensor(S[21,21,21], dtype=torch.float32, requires_grad=True)
tau_tensor = torch.tensor(tau[21,21,21], dtype=torch.float32, requires_grad=True)
delta_tensor = torch.tensor(delta[21,21,21], dtype=torch.float32, requires_grad=True)

epoch = 0
total_loss = 0

I1 = torch.trace(torch.mm(S_tensor,S_tensor))
I2 = torch.trace(torch.mm(R_tensor,R_tensor))
I3 = torch.trace(torch.mm(torch.mm(S_tensor,S_tensor),S_tensor))
I4 = torch.trace(torch.mm(torch.mm(S_tensor,R_tensor),R_tensor))
I5 = torch.trace(torch.mm(torch.mm(S_tensor,S_tensor),torch.mm(R_tensor,R_tensor)))
I6 = torch.trace(torch.mm(torch.mm(torch.mm(S_tensor,S_tensor),torch.mm(R_tensor,R_tensor)),torch.mm(S_tensor,R_tensor)))

# get 6 coefficients for model
input = torch.tensor([I1, I2, I3, I4, I5, I6], dtype=torch.float32)

# Training loop
for i in range(num_epochs):
    # Zero the gradients (reset the gradients for each batch)
    optimizer.zero_grad()

    # Forward pass
    output = model(input)  # Compute the output of the model

    # Compute the predicted SGS stress tensor
    pred = nu_deriv_funct.apply(output,S,delta_tensor,[21,21,21],2*np.pi/64,tau)

    # Compute the loss
    loss = loss_function(pred, dtau_del(tau,delta_tensor,[21,21,21],2*np.pi/64))
    loss_list.append(loss.item())
    total_loss += loss.item()

    print(f'\rEpoch: {epoch}/{num_epochs} | Epoch Progress: {epoch/num_epochs*100:5.2f}% | Loss: {loss.item():10.2f}', end='')

    # Backpropagation
    loss.backward()
    optimizer.step()

    epoch += 1

# Calculating analytical closure term -2 * d_Sij/dxj
grid_spacing = np.pi * 2 / 64

# differentiating tau_ij wrt x_j
dSijdx1 = (S - np.roll(S, 1, axis=0)) / grid_spacing
dSijdx2 = (S - np.roll(S, 1, axis=1)) / grid_spacing
dSijdx3 = (S - np.roll(S, 1, axis=2)) / grid_spacing

# interpolating to local values
dSijdx1 = dSijdx1[21,21,21]
dSijdx2 = dSijdx2[21,21,21]
dSijdx3 = dSijdx3[21,21,21]

dS1jdxj = dSijdx1[0,0] + dSijdx2[0,1] + dSijdx3[0,2]
dS2jdxj = dSijdx1[1,0] + dSijdx2[1,1] + dSijdx3[1,2]
dS3jdxj = dSijdx1[2,0] + dSijdx2[2,1] + dSijdx3[2,2]

delta_local = delta[21,21,21,0,:]
dSijdxj = np.array([dS1jdxj, dS2jdxj, dS3jdxj])

# Defining x,y,z coordinates in numpy array
dx = dy = dz = 2*np.pi/64
Nx = Ny = Nz = 64
x = np.reshape(np.linspace(dx/2,(Nx-1/2)*dx,Nx), (Nx, 1, 1))
y = np.reshape(np.linspace(dy/2,(Ny-1/2)*dy,Ny), (1, Ny, 1))
z = np.reshape(np.linspace(dz/2,(Nz-1/2)*dz,Nz), (1, 1, Nz))
x = np.tile(x, (1,Ny,Nz))
y = np.tile(y, (Nx,1,Nz))
z = np.tile(z, (Nx,Ny,1))

# Calculating analytical inputs at local point -> should all be zero
print(f'\nInput node vector:\n{input.detach().numpy()}')

# Print predicted and actual closure terms
print(f'Predicted closure term: {pred.detach().numpy()}')
print(f'Closure from nu = 1: {nu_deriv_funct.apply(torch.tensor(1, dtype=torch.float32),S,delta_tensor,[21,21,21],2*np.pi/64,tau).detach().numpy()}')
print(f'Analytical closure term: {delta_local + 2 * dSijdxj}')
print(f'Actual closure term: {dtau_del(tau,delta_tensor,[21,21,21],2*np.pi/64).detach().numpy()}')

# Test the backwarding function
delta_test = torch.zeros([3,1])
grad_output_test = torch.tensor([1,2,3], dtype=torch.float32)
dnudclose = -2 * dtau_del(S, delta_test, [21,21,21], grid_spacing)
grad_verify = grad_output_test * dnudclose
print(f'Backwarding function unsummed test: {dnudclose.detach().numpy()}')
print(f'Backwarding function test: {grad_verify.sum().view(1)}')

# Print predicted and actual SGS stress tensors
print(f'Actual SGS stress tensor:\n{tau_tensor.detach().numpy()}')
print(f'Actual S tensor:\n{S_tensor.detach().numpy()}')

# Print loss from nu_t = 1, -2*S_ij vs tau_ij (~0)
print(f'Loss from nu_t = 1: {loss_function(-2*S_tensor, tau_tensor).item()}')

conv_size = 1
plt.plot(np.convolve(loss_list, np.ones(conv_size), 'valid') / conv_size, label='Training')
plt.title('Loss vs. Iteration (inconsistent model)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
plt.close()