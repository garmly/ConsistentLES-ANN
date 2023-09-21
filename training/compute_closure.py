import numpy as np
import torch

# output dtau / dx_j + delta_ij = -2 d(nu_t * S_ij) / dx_j
def dtau_del(tau, delta, index, grid_spacing):
        
        # dtau_1j / dx_j
        dtau1jdxj = (tau - np.roll(tau, 1, axis=0)) / grid_spacing
        dtau1jdxj = dtau1jdxj[index][0,:]
        dtau2jdxj = (tau - np.roll(tau, 1, axis=1)) / grid_spacing
        dtau2jdxj = dtau2jdxj[index][1,:]
        dtau3jdxj = (tau - np.roll(tau, 1, axis=2)) / grid_spacing
        dtau3jdxj = dtau3jdxj[index][2,:]

        dtauijdxj = [dtau1jdxj, dtau2jdxj, dtau3jdxj]
        print(dtauijdxj.shape)

        dtau = dtauijdxj + delta

        return torch.tensor(dtau, dtype=torch.float32)

# reform dtau_del as a 
class nu_deriv_funct(torch.autograd.Function):
        def forward(ctx, input_data, delta):
                pass
        def backward():
                pass