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

def read_SGS_binary_TEST(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")
    
    with open(filename, 'rb') as binfile:
        Nx = np.fromfile(binfile, dtype='>i4', count=1)[0]
        x = np.fromfile(binfile, dtype='>f8', count=Nx)
        Ny = np.fromfile(binfile, dtype='>i4', count=1)[0]
        y = np.fromfile(binfile, dtype='>f8', count=Ny)
        Nz = np.fromfile(binfile, dtype='>i4', count=1)[0]
        z = np.fromfile(binfile, dtype='>f8', count=Nz)

        data = np.zeros((Nx, Ny, Nz, 3, 3), dtype='f8')
        NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
        uu = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
        NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
        uv = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
        NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
        uw = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
        NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
        vu = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
        NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
        vv = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
        NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
        vw = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
        NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
        wu = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
        NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
        wv = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
        NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
        ww = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
        
        data[:, :, :, 0, 0] = uu
        data[:, :, :, 0, 1] = uv
        data[:, :, :, 0, 2] = uw
        data[:, :, :, 1, 0] = vu
        data[:, :, :, 1, 1] = vv
        data[:, :, :, 1, 2] = vw
        data[:, :, :, 2, 0] = wu
        data[:, :, :, 2, 1] = wv
        data[:, :, :, 2, 2] = ww

    return data

# Define coordinate and grid
Nx = Ny = Nz = 64
i = [21,21,21]

# Read data from .bin files
Tau = read_SGS_binary_TEST(f"{path}/filtered/SGS/Tau/TEST.bin")
R = read_SGS_binary_TEST(f"{path}/filtered/SGS/R/TEST.bin")
S = read_SGS_binary_TEST(f"{path}/filtered/SGS/S/TEST.bin")
delta = read_delta(f"{path}/filtered/delta/TEST.bin")

# Generate the tensors at the specific point in the grid
R_tensor = torch.tensor(R[i[0],i[1],i[2]], dtype=torch.float32, requires_grad=True)
S_tensor = torch.tensor(S[i[0],i[1],i[2]], dtype=torch.float32, requires_grad=True)
Tau_tensor = torch.tensor(Tau[i[0],i[1],i[2]], dtype=torch.float32, requires_grad=True)
Delta_tensor = torch.tensor(delta[i[0],i[1],i[2]], dtype=torch.float32, requires_grad=True)

# Calculate 6 scalar inputs for the specific point in the file
I1 = torch.trace(torch.mm(S_tensor,S_tensor))
I2 = torch.trace(torch.mm(R_tensor,R_tensor))
I3 = torch.trace(torch.mm(torch.mm(S_tensor,S_tensor),S_tensor))
I4 = torch.trace(torch.mm(torch.mm(S_tensor,R_tensor),R_tensor))
I5 = torch.trace(torch.mm(torch.mm(S_tensor,S_tensor),torch.mm(R_tensor,R_tensor)))
I6 = torch.trace(torch.mm(torch.mm(torch.mm(S_tensor,S_tensor),torch.mm(R_tensor,R_tensor)),torch.mm(S_tensor,R_tensor)))

# get 6 coefficients for model
inputs = torch.tensor([I1, I2, I3, I4, I5, I6], dtype=torch.float32)

# Get the immediate neighbors of S and Tau at point i with periodic boundary conditions
S_neighbors = np.zeros([3,3,3,3,3])
Tau_neighbors = np.zeros([3,3,3,3,3])
S_neighbors[0,0,0] = S[(i[0]-1)%Nx,(i[1]-1)%Ny,(i[2]-1)%Nz]
S_neighbors[0,0,1] = S[(i[0]-1)%Nx,(i[1]-1)%Ny,(i[2])%Nz]
S_neighbors[0,0,2] = S[(i[0]-1)%Nx,(i[1]-1)%Ny,(i[2]+1)%Nz]
S_neighbors[0,1,0] = S[(i[0]-1)%Nx,(i[1])%Ny,(i[2]-1)%Nz]
S_neighbors[0,1,1] = S[(i[0]-1)%Nx,(i[1])%Ny,(i[2])%Nz]
S_neighbors[0,1,2] = S[(i[0]-1)%Nx,(i[1])%Ny,(i[2]+1)%Nz]
S_neighbors[0,2,0] = S[(i[0]-1)%Nx,(i[1]+1)%Ny,(i[2]-1)%Nz]
S_neighbors[0,2,1] = S[(i[0]-1)%Nx,(i[1]+1)%Ny,(i[2])%Nz]
S_neighbors[0,2,2] = S[(i[0]-1)%Nx,(i[1]+1)%Ny,(i[2]+1)%Nz]
S_neighbors[1,0,0] = S[(i[0])%Nx,(i[1]-1)%Ny,(i[2]-1)%Nz]
S_neighbors[1,0,1] = S[(i[0])%Nx,(i[1]-1)%Ny,(i[2])%Nz]
S_neighbors[1,0,2] = S[(i[0])%Nx,(i[1]-1)%Ny,(i[2]+1)%Nz]
S_neighbors[1,1,0] = S[(i[0])%Nx,(i[1])%Ny,(i[2]-1)%Nz]
S_neighbors[1,1,1] = S[(i[0])%Nx,(i[1])%Ny,(i[2])%Nz]
S_neighbors[1,1,2] = S[(i[0])%Nx,(i[1])%Ny,(i[2]+1)%Nz]
S_neighbors[1,2,0] = S[(i[0])%Nx,(i[1]+1)%Ny,(i[2]-1)%Nz]
S_neighbors[1,2,1] = S[(i[0])%Nx,(i[1]+1)%Ny,(i[2])%Nz]
S_neighbors[1,2,2] = S[(i[0])%Nx,(i[1]+1)%Ny,(i[2]+1)%Nz]
S_neighbors[2,0,0] = S[(i[0]+1)%Nx,(i[1]-1)%Ny,(i[2]-1)%Nz]
S_neighbors[2,0,1] = S[(i[0]+1)%Nx,(i[1]-1)%Ny,(i[2])%Nz]
S_neighbors[2,0,2] = S[(i[0]+1)%Nx,(i[1]-1)%Ny,(i[2]+1)%Nz]
S_neighbors[2,1,0] = S[(i[0]+1)%Nx,(i[1])%Ny,(i[2]-1)%Nz]
S_neighbors[2,1,1] = S[(i[0]+1)%Nx,(i[1])%Ny,(i[2])%Nz]
S_neighbors[2,1,2] = S[(i[0]+1)%Nx,(i[1])%Ny,(i[2]+1)%Nz]
S_neighbors[2,2,0] = S[(i[0]+1)%Nx,(i[1]+1)%Ny,(i[2]-1)%Nz]
S_neighbors[2,2,1] = S[(i[0]+1)%Nx,(i[1]+1)%Ny,(i[2])%Nz]
S_neighbors[2,2,2] = S[(i[0]+1)%Nx,(i[1]+1)%Ny,(i[2]+1)%Nz]
Tau_neighbors[0,0,0] = Tau[(i[0]-1)%Nx,(i[1]-1)%Ny,(i[2]-1)%Nz]
Tau_neighbors[0,0,1] = Tau[(i[0]-1)%Nx,(i[1]-1)%Ny,(i[2])%Nz]
Tau_neighbors[0,0,2] = Tau[(i[0]-1)%Nx,(i[1]-1)%Ny,(i[2]+1)%Nz]
Tau_neighbors[0,1,0] = Tau[(i[0]-1)%Nx,(i[1])%Ny,(i[2]-1)%Nz]
Tau_neighbors[0,1,1] = Tau[(i[0]-1)%Nx,(i[1])%Ny,(i[2])%Nz]
Tau_neighbors[0,1,2] = Tau[(i[0]-1)%Nx,(i[1])%Ny,(i[2]+1)%Nz]
Tau_neighbors[0,2,0] = Tau[(i[0]-1)%Nx,(i[1]+1)%Ny,(i[2]-1)%Nz]
Tau_neighbors[0,2,1] = Tau[(i[0]-1)%Nx,(i[1]+1)%Ny,(i[2])%Nz]
Tau_neighbors[0,2,2] = Tau[(i[0]-1)%Nx,(i[1]+1)%Ny,(i[2]+1)%Nz]
Tau_neighbors[1,0,0] = Tau[(i[0])%Nx,(i[1]-1)%Ny,(i[2]-1)%Nz]
Tau_neighbors[1,0,1] = Tau[(i[0])%Nx,(i[1]-1)%Ny,(i[2])%Nz]
Tau_neighbors[1,0,2] = Tau[(i[0])%Nx,(i[1]-1)%Ny,(i[2]+1)%Nz]
Tau_neighbors[1,1,0] = Tau[(i[0])%Nx,(i[1])%Ny,(i[2]-1)%Nz]
Tau_neighbors[1,1,1] = Tau[(i[0])%Nx,(i[1])%Ny,(i[2])%Nz]
Tau_neighbors[1,1,2] = Tau[(i[0])%Nx,(i[1])%Ny,(i[2]+1)%Nz]
Tau_neighbors[1,2,0] = Tau[(i[0])%Nx,(i[1]+1)%Ny,(i[2]-1)%Nz]
Tau_neighbors[1,2,1] = Tau[(i[0])%Nx,(i[1]+1)%Ny,(i[2])%Nz]
Tau_neighbors[1,2,2] = Tau[(i[0])%Nx,(i[1]+1)%Ny,(i[2]+1)%Nz]
Tau_neighbors[2,0,0] = Tau[(i[0]+1)%Nx,(i[1]-1)%Ny,(i[2]-1)%Nz]
Tau_neighbors[2,0,1] = Tau[(i[0]+1)%Nx,(i[1]-1)%Ny,(i[2])%Nz]
Tau_neighbors[2,0,2] = Tau[(i[0]+1)%Nx,(i[1]-1)%Ny,(i[2]+1)%Nz]
Tau_neighbors[2,1,0] = Tau[(i[0]+1)%Nx,(i[1])%Ny,(i[2]-1)%Nz]
Tau_neighbors[2,1,1] = Tau[(i[0]+1)%Nx,(i[1])%Ny,(i[2])%Nz]
Tau_neighbors[2,1,2] = Tau[(i[0]+1)%Nx,(i[1])%Ny,(i[2]+1)%Nz]
Tau_neighbors[2,2,0] = Tau[(i[0]+1)%Nx,(i[1]+1)%Ny,(i[2]-1)%Nz]
Tau_neighbors[2,2,1] = Tau[(i[0]+1)%Nx,(i[1]+1)%Ny,(i[2])%Nz]
Tau_neighbors[2,2,2] = Tau[(i[0]+1)%Nx,(i[1]+1)%Ny,(i[2]+1)%Nz]

# Calculate the target data for the specific point in the file
target = dtau_del(Tau_neighbors,Delta_tensor,2*np.pi/64)

# Training loop
for epoch in range(num_epochs):
    # Training loss
    model.train()
    training_loss = 0.0
    epoch_counter = 0

    # Zero the gradients (reset the gradients for each batch)
    optimizer.zero_grad()

    delta_tensor = Delta_tensor.view(1,3,3)
    S_local = S
    tau = Tau
    input = inputs.view(1,6)

    # Forward pass
    output = model(input)  # Compute the output of the model

    # Compute the predicted SGS stress tensor
    pred = nu_deriv_funct.apply(output,S,delta_tensor,2*np.pi/64,tau)

    # Compute the loss
    loss = loss_function(pred, target)
    training_loss += loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    epoch_counter += 1
    print(f'\rEpoch: {epoch}/{num_epochs} | Epoch Progress: {epoch/num_epochs*100:5.2f}%  | Loss (Training): {training_loss/epoch_counter:10.2f}', end='')
    

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
print(f'Closure from nu = 1: {nu_deriv_funct.apply(torch.tensor(1, dtype=torch.float32),S,delta_tensor,[21,21,21],2*np.pi/64,tau).detach().numpy()}')
print(f'Analytical closure term: {delta_local + 2 * dSijdxj}')
print(f'Actual closure term: {dtau_del(tau,delta_tensor,[21,21,21],2*np.pi/64).detach().numpy()}')
print(f'Predicted closure term: {pred.detach().numpy()}')

# Test the backwarding function
delta_test = torch.zeros([3,1])
grad_output_test = torch.tensor([1,2,3], dtype=torch.float32)
dnudclose = -2 * dtau_del(S, delta_test, [21,21,21], grid_spacing)
grad_verify = grad_output_test * dnudclose
print(f'Backwarding function unsummed test: {dnudclose.detach().numpy()}')
print(f'Backwarding function test: {grad_verify.sum().view(1)}')

# Print predicted and actual SGS stress tensors
print(f'Actual SGS stress tensor:\n{tau[i].detach().numpy()}')
print(f'Actual S tensor:\n{S_tensor.detach().numpy()}')

# Print loss from nu_t = 1, -2*S_ij vs tau_ij (~0)
print(f'Loss from nu_t = 1: {loss_function(-2*S_tensor, tau[i]).item()}')

conv_size = 1
plt.plot(np.convolve(loss_list, np.ones(conv_size), 'valid') / conv_size, label='Training')
plt.title('Loss vs. Iteration (inconsistent model)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
plt.close()