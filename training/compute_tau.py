import numpy as np

def compute_tau(R,S,grid_spacing,C):
    # compute the SGS stress tensor from R, S, and coefficients C
    def single_tau(Rtens,Stens):
        tau = np.empty([3,3])
        tau = C[0] * grid_spacing**2 * np.sqrt(np.trace(Stens*Stens)) * Stens
        tau += C[1] * grid_spacing**2 * (Stens*Stens - np.diag(Stens*Stens))
        tau += C[2] * grid_spacing**2 * (Rtens*Rtens - np.diag(Rtens*Rtens))
        return tau

    tau = np.empty([R.shape[0], R.shape[1], R.shape[2], 3, 3])
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            for k in range(R.shape[2]):
                tau[i,j,k] = single_tau(R[i,j,k], S[i,j,k])
    return tau
