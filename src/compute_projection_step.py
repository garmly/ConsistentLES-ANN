import numpy as np
from pyfftw.interfaces import scipy_fftpack as fftw
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
    
    phat = fftw.fftn(grid.p)

    kk = grid.kxx[:, np.newaxis, np.newaxis] + \
         grid.kyy[np.newaxis, :, np.newaxis] + \
         grid.kzz[np.newaxis, np.newaxis, :]
    kk[0,0,0] = 1
    phat /= kk
    
    grid.p = fftw.ifftn(phat).real

    u -= (grid.p - roll_view(grid.p,1,axis=0)) / grid.dx
    v -= (grid.p - roll_view(grid.p,1,axis=1)) / grid.dy
    w -= (grid.p - roll_view(grid.p,1,axis=2)) / grid.dz
    
    return u,v,w