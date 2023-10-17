import numpy as np
import torch
import copy as cp

# output dtau / dx_j + delta_ij = -2 d(nu_t * S_ij) / dx_j
def dtau_del(tau, delta, index, grid_spacing):

        # differentiating tau_ij wrt x_j
        dtauijdx1 = (tau - np.roll(tau, 1, axis=0)) / grid_spacing
        dtauijdx2 = (tau - np.roll(tau, 1, axis=1)) / grid_spacing
        dtauijdx3 = (tau - np.roll(tau, 1, axis=2)) / grid_spacing

        # interpolating to local values
        dtauijdx1 = dtauijdx1[index[0], index[1], index[2]]
        dtauijdx2 = dtauijdx2[index[0], index[1], index[2]]
        dtauijdx3 = dtauijdx3[index[0], index[1], index[2]]

        dtau1jdxj = dtauijdx1[0,0] + dtauijdx2[0,1] + dtauijdx3[0,2]
        dtau2jdxj = dtauijdx1[1,0] + dtauijdx2[1,1] + dtauijdx3[1,2]
        dtau3jdxj = dtauijdx1[2,0] + dtauijdx2[2,1] + dtauijdx3[2,2]

        dtauijdxj = [dtau1jdxj, dtau2jdxj, dtau3jdxj]

        # resolved closure term
        dtau = delta[0,:].detach().numpy() - dtauijdxj

        return torch.tensor(dtau, dtype=torch.float32)

# reform dtau_del as a torch function
class nu_deriv_funct(torch.autograd.Function):
        def forward(ctx, input_data, tau, delta, index, grid_spacing):
                pred_tau = cp.deepcopy(tau)
                pred_tau[index] = input_data.detach().numpy()
                return dtau_del(pred_tau, delta, index, grid_spacing)
        
        def backward(ctx, grad_output):
                # d closure_k / d tau_ij
                grad = torch.tile(grad_output/3, (3,1))
                return grad, None, None, None, None