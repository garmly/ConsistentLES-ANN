import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
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
        self.layer1 = nn.Linear(15, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 20)
        self.layer4 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 9)

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
num_epochs = 1

# Number of snapshots
num_batches = 1

loss_list = []
param6 = False
commute = True
validation = False

# Training loop
for epoch in range(num_epochs+1):
    total_loss = 0.0

    for batch_num in range(num_batches):
        # Load the data
        tau = read_SGS(f"./out/filtered/SGS/Tau/t{batch_num + 1}.csv", 64, 64, 64)
        R = read_SGS(f"./out/filtered/SGS/R/t{batch_num + 1}.csv", 64, 64, 64)
        S = read_SGS(f"./out/filtered/SGS/S/t{batch_num + 1}.csv", 64, 64, 64)
        delta = read_SGS(f"./out/filtered/delta/t{batch_num + 1}.csv", 64, 64, 64)

        element = 0
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                for k in range(R.shape[2]):
                    # Convert the data to tensors
                    R_tensor = torch.tensor(R[i,j,k], dtype=torch.float32, requires_grad=True)
                    S_tensor = torch.tensor(S[i,j,k], dtype=torch.float32, requires_grad=True)
                    tau_tensor = torch.tensor(tau[i,j,k], dtype=torch.float32, requires_grad=True)
                    delta_tensor = torch.tensor(delta[i,j,k], dtype=torch.float32, requires_grad=True)

                    # Zero the gradients (reset the gradients for each batch)

                    optimizer.zero_grad()

                    if param6:
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
                    else:
                        in1 = torch.flatten(R_tensor)
                        in2 = torch.flatten(torch.tril(S_tensor))
                        in2 = in2[in2 != 0]
                        input = torch.tensor(torch.cat((in1, in2)),
                                             dtype=torch.float32, requires_grad=True)

                    # Forward pass
                    output = model(input)  # Compute the output of the model

                    # Compute the predicted SGS stress tensor
                    if param6:
                        pred = tau_6c_funct.apply(output, S_tensor)
                    elif commute:
                        pred = output
                    else:
                        pred = tau_4c_funct.apply(output, R_tensor, S_tensor, 2*np.pi/256)

                    # Compute the loss
                    loss = loss_function(pred, torch.flatten(delta_tensor))
                    loss_list.append(loss.item())
                    total_loss += loss.item()

                    print(f'\rBatch: {batch_num+1}/{num_batches} | Batch Progress: {element/64**3*100:5.2f}% | Loss: {loss.item():10.4f}', end='')

                    # Backpropagation
                    loss.backward()

                    # Update the model's parameters
                    if not validation:
                        optimizer.step()

                    element += 1

    print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / num_batches / (64**3):.4f}, Std Dev: {np.std(loss_list):.4f}")

    if epoch == num_epochs-1:
        validation = True
        training_list = loss_list
        loss_list = []

plt.plot(np.convolve(training_list, np.ones(10000), 'valid') / 10000, label='Training', linestyle='--')
plt.plot(np.convolve(loss_list, np.ones(10000), 'valid') / 10000, label='Validation')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
plt.close()
