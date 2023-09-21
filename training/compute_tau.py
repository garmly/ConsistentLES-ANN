import numpy as np
import torch

class tau_3c_funct(torch.autograd.Function):
    def forward(ctx, input_data, R,S, grid_spacing):
        ctx.save_for_backward(input_data, R,S)
        R = R.detach().numpy()
        S = S.detach().numpy()
        ctx.grid_spacing = grid_spacing
        tau = single_tau(input_data.detach().numpy(), R, S, grid_spacing)
        return torch.tensor(tau, dtype=torch.float32)

    def backward(ctx, grad_output):
        input_data, R,S = ctx.saved_tensors
        grid_spacing = ctx.grid_spacing
        R = R.detach().numpy()
        S = S.detach().numpy()
        SbarS = Sstar = Rstar = np.empty([3, 3])
        SbarS = np.sqrt(np.trace(S*S)) * S
        Sstar = 1/3 * np.diag(np.sum(S*S, axis=1))
        Rstar = 1/3 * np.diag(np.sum(R*R, axis=1))
        grad_input = [SbarS, Sstar, Rstar] * grad_output.detach().numpy() * grid_spacing**2
        grad_input = grad_input.sum(axis=-1)
        grad_input = torch.tensor(grad_input, dtype=torch.float32)
        return grad_input, None, None, None

def single_tau(C, Rtens,Stens, grid_spacing):
    tau = np.empty([3,3])
    tau = C[0] * grid_spacing**2 * np.sqrt(np.trace(Stens*Stens)) * Stens
    tau += C[1] * grid_spacing**2 * 1/3 * np.diag(np.sum(Stens*Stens, axis=1))
    tau += C[2] * grid_spacing**2 * 1/3 * np.diag(np.sum(Rtens*Rtens, axis=1))
    return tau

class tau_nu_funct(torch.autograd.Function):
    def forward(ctx, input_data, S):
        ctx.save_for_backward(S)
        tau = -2 * input_data * S
        return tau
    
    def backward(ctx, grad_output):
        S = ctx.saved_tensors[0]
        grad_input = -2 * S * grad_output
        return [grad_input.sum()], None