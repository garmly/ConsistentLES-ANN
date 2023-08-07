import numpy as np
import torch

class tau_funct(torch.autograd.Function):
    def forward(ctx, input_data, R,S, grid_spacing):
        ctx.save_for_backward(input_data, R,S)
        ctx.grid_spacing = grid_spacing
        # compute the SGS stress tensor from R, S, and coefficients C
        R = R.detach().numpy()
        S = S.detach().numpy()
        tau = np.empty([R.shape[0], R.shape[1], R.shape[2], 3, 3])
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                for k in range(R.shape[2]):
                    tau[i,j,k] = single_tau(input_data.detach().numpy(), R[i,j,k], S[i,j,k], grid_spacing)
        
        return torch.tensor(tau, dtype=torch.float32)

    def backward(ctx, grad_output):
        input_data, R,S = ctx.saved_tensors
        grid_spacing = ctx.grid_spacing
        R = R.detach().numpy()
        S = S.detach().numpy()
        SbarS = Sstar = Rstar = np.empty([R.shape[0], R.shape[1], R.shape[2], 3, 3])
        for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    for k in range(R.shape[2]):
                        SbarS[i,j,k] = np.sqrt(np.trace(S[i,j,k]*S[i,j,k])) * S[i,j,k]
                        Sstar[i,j,k] = 1/3 * np.diag(np.sum(S[i,j,k]*S[i,j,k], axis=1))
                        Rstar[i,j,k] = 1/3 * np.diag(np.sum(R[i,j,k]*R[i,j,k], axis=1))
        grad_input = [SbarS, Sstar, Rstar] * grad_output.detach().numpy() * grid_spacing**2
        grad_input = grad_input.sum(axis=(1,2,3,4,5))
        grad_input = torch.tensor(grad_input, dtype=torch.float32)
        return grad_input, None, None, None

def single_tau(C, Rtens,Stens, grid_spacing):
    tau = np.empty([3,3])
    tau = C[0] * grid_spacing**2 * np.sqrt(np.trace(Stens*Stens)) * Stens
    tau += C[1] * grid_spacing**2 * 1/3 * np.diag(np.sum(Stens*Stens, axis=1))
    tau += C[2] * grid_spacing**2 * 1/3 * np.diag(np.sum(Rtens*Rtens, axis=1))
    return tau