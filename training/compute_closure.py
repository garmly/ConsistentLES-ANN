import numpy as np
import torch

# output dtau / dx_j + delta_ij = -2 d(nu_t * S_ij) / dx_j
def dtau_del(tau, delta, index, grid_spacing):
        
        # dtau_1j / dx_j
        dtau1jdxj = (tau[index][1,:] - np.roll(tau, 1, axis=0)) / grid_spacing
        dtau1jdxj += (tau[index][1,:] - np.roll(tau, 1, axis=0)) / grid_spacing
        dtau1jdxj += (tau[index][1,:] - np.roll(tau, 1, axis=0)) / grid_spacing

        # dtau_2j / dx_j
        dtau2jdxj = (tau[index][2,:] - np.roll(tau, 1, axis=1)) / grid_spacing
        dtau2jdxj += (tau[index][2,:] - np.roll(tau, 1, axis=1)) / grid_spacing
        dtau2jdxj += (tau[index][2,:] - np.roll(tau, 1, axis=1)) / grid_spacing

        # dtau_3j / dx_j
        dtau3jdxj = (tau[index][3,:] - np.roll(tau, 1, axis=2)) / grid_spacing
        dtau3jdxj += (tau[index][3,:] - np.roll(tau, 1, axis=2)) / grid_spacing
        dtau3jdxj += (tau[index][3,:] - np.roll(tau, 1, axis=2)) / grid_spacing

        dtauijdxj = np.array([dtau1jdxj, dtau2jdxj, dtau3jdxj])
        dtau = dtauijdxj + delta

        return torch.tensor(dtau, dtype=torch.float32)

