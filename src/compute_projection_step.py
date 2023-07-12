import numpy as np
import scipy.fft
from src.roll_view import *

def compute_projection_step(grid, principal=True): # if True, use u,v,w - if False, Fu,Fv,Fw
    if (principal):
        u = grid.u
        v = grid.v
        w = grid.w
    else:
        u = grid.Fu
        v = grid.Fv
        w = grid.Fw

    # compute divergence of u,v,w
    grid.p = (roll_view(u,-1,axis=0) - u) / grid.dx + \
             (roll_view(v,-1,axis=1) - v) / grid.dy + \
             (roll_view(w,-1,axis=2) - w) / grid.dz
    
    phat = scipy.fft.fftn(grid.p)

    for index, value in np.ndenumerate(grid.p):
        kk = grid.kxx[index[0]] + grid.kyy[index[1]] + grid.kzz[index[2]]
        if (not np.array_equal(index, [0,0,0])):
            phat[index] /= kk
    
    grid.p = scipy.fft.ifftn(phat).real

    u -= (grid.p - roll_view(grid.p,1,axis=0)) / grid.dx
    v -= (grid.p - roll_view(grid.p,1,axis=1)) / grid.dy
    w -= (grid.p - roll_view(grid.p,1,axis=2)) / grid.dz
    
    return u,v,w