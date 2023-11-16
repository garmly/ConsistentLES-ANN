import torch
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
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
        super().__init__()
        self.requires_grad_(True)
        self.layer1 = nn.Linear(6, 50)
        self.layer2 = nn.Linear(50, 50)
        self.output_layer = nn.Linear(50, 1)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.output_layer(x)
        return x

class LESDataset(Dataset):
    def __init__(self, data_dir, Nx, Ny, Nz, fileno, step=1):
        self.data_dir = data_dir
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.fileno = fileno
        self.step = step
        self.R = np.zeros([fileno,Nx,Ny,Nz,3,3])
        self.S = np.zeros([fileno,Nx,Ny,Nz,3,3])
        self.Tau = np.zeros([fileno,Nx,Ny,Nz,3,3])
        self.Delta = np.zeros([fileno,Nx,Ny,Nz,3,3])

        for i in range(fileno):
            R_i = read_SGS_binary(f'{self.data_dir}SGS/R/t{i+1}.bin')[...,0]
            S_i = read_SGS_binary(f'{self.data_dir}SGS/S/t{i+1}.bin')[...,0]
            Tau_i = read_SGS_binary(f'{self.data_dir}SGS/Tau/t{i+1}.bin')[...,0]
            Delta_i = read_delta(f'{self.data_dir}delta/t{i+1}.bin')
            self.R[i,...] = R_i
            self.S[i,...] = S_i
            self.Tau[i,...] = Tau_i
            self.Delta[i,...] = Delta_i

    def __len__(self):
        return self.fileno * (self.Nx * self.Ny * self.Nz) // self.step

    def __getitem__(self, idx):
        idx = (idx * self.step) % self.__len__()
        file_idx = idx // (self.Nx * self.Ny * self.Nz)
        point_idx = idx % (self.Nx * self.Ny * self.Nz)

        R = self.R[file_idx,...]
        S = self.S[file_idx,...]
        Tau = self.Tau[file_idx,...]
        Delta = self.Delta[file_idx,...]

        # convert point_idx to a coordinate in the grid
        i = np.zeros(3)
        i[0] = point_idx // (self.Ny * self.Nz)
        i[1] = (point_idx % (self.Ny * self.Nz)) // self.Nz
        i[2] = point_idx % self.Nz
        i = [int(i[0]), int(i[1]), int(i[2])]

        # Generate the tensors at the specific point in the grid
        R_tensor = torch.tensor(R[i[0],i[1],i[2]], dtype=torch.float32, requires_grad=True)
        S_tensor = torch.tensor(S[i[0],i[1],i[2]], dtype=torch.float32, requires_grad=True)
        Tau_tensor = torch.tensor(Tau[i[0],i[1],i[2]], dtype=torch.float32, requires_grad=True)
        Delta_tensor = torch.tensor(Delta[i[0],i[1],i[2]], dtype=torch.float32, requires_grad=True)

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
        S_neighbors = np.zeros([3,3,3,3,3])
        Tau_neighbors = np.zeros([3,3,3,3,3])

        # Get Tau and S neghbors
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    S_neighbors[a,b,c] = S[(i[0]+a-1)%self.Nx,(i[1]+b-1)%self.Ny,(i[2]+c-1)%self.Nz]
                    Tau_neighbors[a,b,c] = Tau[(i[0]+a-1)%self.Nx,(i[1]+b-1)%self.Ny,(i[2]+c-1)%self.Nz]

        # Calculate the target data for the specific point in the file
        target = dtau_del(Tau_neighbors,Delta_tensor,2*np.pi/64)

        # Check if shapes are correct
        assert inputs.shape == (6,)
        assert target.shape == (3,)
        assert S_neighbors.shape == (3,3,3,3,3)
        assert Tau_neighbors.shape == (3,3,3,3,3)
        assert Delta_tensor.shape == (3,3)

        return {
            'inputs': inputs,
            'target': target,
            'S_neighbors': S_neighbors,
            'Tau_neighbors': Tau_neighbors,
            'delta': Delta_tensor
        }

# Create data loader
batch_size = 1
full_dataset = LESDataset('./in/filtered/', 64, 64, 64, 8, 700)

# Split the data into training and validation datasets
validation_size = 0.2
num_validation = int(validation_size * len(full_dataset))
num_training = len(full_dataset) - num_validation
train_dataset, validation_dataset = random_split(full_dataset, [num_training, num_validation])

print(f'Training dataset size: {num_training}')
print(f'Validation dataset size: {num_validation}')
print(f'Total dataset size: {len(full_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Initialize the neural network
model = SGS_ANN()

# Define the loss function (cost function)
loss_function = nn.MSELoss()

# Define the optimizer as stochastic gradient descent (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)

# Number of training epochs
num_epochs = 20
epochs_conv = 0

print('Starting training...')
validation_loss_list = []
training_loss_list = []
for epoch in range(num_epochs):
    # Training loss
    model.train(True)
    training_loss = 0.0
    epoch_counter = 0
    for batch in train_loader:
        # Zero the gradients (reset the gradients for each batch)
        optimizer.zero_grad()

        delta_tensor = batch['delta'][0,...]
        S = batch['S_neighbors'][0,...].detach().numpy()
        tau = batch['Tau_neighbors'][0,...].detach().numpy()
        input = batch['inputs'][0,...]

        # Forward pass
        output = model(input)  # Compute the output of the model

        # Compute the predicted SGS stress tensor
        pred = nu_deriv_funct.apply(output,S,delta_tensor,2*np.pi/64,tau)

        # Compute the loss
        loss = loss_function(pred, batch['target'][0,...])
        training_loss += loss.item()

        # Backpropagation
        loss.backward()

        # Clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 500)

        optimizer.step()
        
        epoch_counter += 1
        print(f'\rEpoch: {epoch}/{num_epochs} | Epoch Progress: {epoch/num_epochs*100:5.2f}%  | Loss (Training): {training_loss/epoch_counter:10.2f}', end='')
    
    # Validation loss
    model.eval()
    validation_loss = 0.0
    epoch_counter = 0
    with torch.no_grad():
        for batch in validation_loader:
            delta_tensor = batch['delta'][0,...]
            S = batch['S_neighbors'][0,...].detach().numpy()
            tau = batch['Tau_neighbors'][0,...].detach().numpy()
            input = batch['inputs'][0,...]
            pred = model(input)
            loss = loss_function(pred, batch['target'][0,...])
            validation_loss += loss.item()
            epoch_counter += 1
            print(f'\rEpoch: {epoch}/{num_epochs} | Epoch Progress: {epoch/num_epochs*100:5.2f}%  | Loss (Validation): {validation_loss/epoch_counter:10.2f}', end='')
    print()

    if epoch % 20 == 0:
        torch.save(model.state_dict(), f'./out/SGS_ANN_{epoch}.pth')
        with open(f'./out/ANN_loss.bin', 'wb') as f:
            np.array([validation_loss_list, training_loss_list]).tofile(f)
    
    if validation_loss / len(validation_loader) > training_loss / len(train_loader):
        epochs_conv += 1
    else:
        epochs_conv = 0

    if epochs_conv == 20:
        print(f'\nTraining converged after {epoch} epochs.')
        break
    
    validation_loss_list.append(validation_loss / len(validation_loader))
    training_loss_list.append(training_loss / len(train_loader))

    if epoch == num_epochs - 1:
        print(f'\nTraining did not converge after {epoch} epochs.')

    epoch += 1

print("\nFinished training!")
plt.plot(validation_loss_list, label='Average Validation Loss')
plt.plot(training_loss_list, label='Average Training Loss')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./out/loss.png')

# Save the model
torch.save(model.state_dict(), './out/SGS_ANN.pth')