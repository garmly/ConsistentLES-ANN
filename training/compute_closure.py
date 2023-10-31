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
# to eliminate compute_tau.py, we instead compute the closure term directly from the eddy viscosity
class nu_deriv_funct(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_data, S, delta, index, grid_spacing, tau_act):
                tau = -2 * input_data * S[index[0], index[1], index[2]]
                ctx.save_for_backward(torch.tensor(index, dtype=torch.int), torch.tensor(grid_spacing, dtype=torch.float32), torch.tensor(S, dtype=torch.float32))
                pred_tau = cp.deepcopy(tau_act)
                pred_tau[index[0], index[1], index[2]] = tau
                return dtau_del(pred_tau, delta, index, grid_spacing)
        
        @staticmethod
        def backward(ctx, grad_output):
                # d closure_k / d tau_ij
                index = ctx.saved_tensors[0].detach().numpy()
                grid_spacing = ctx.saved_tensors[1].detach().numpy()
                S = ctx.saved_tensors[2]
                delta = torch.zeros([3,1])
                dnudclose = -2 * dtau_del(S, delta, index, grid_spacing)
                grad = grad_output * dnudclose
                grad = grad.sum().view(1)
                return grad, None, None, None, None, None
