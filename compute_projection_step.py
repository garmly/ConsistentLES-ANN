import numpy as np
def compute_projection_step(grid, principal=True): # if True, use u,v,p - if False, Fu,Fv,Fw
    if (principal):
        u = grid.u
        v = grid.v
        w = grid.w
    else:
        u = grid.Fu
        v = grid.Fv
        w = grid.Fw

    # compute divergence of u,v,w
    div = (np.roll(u,-1,axis=0) - u) / grid.dx + \
          (np.roll(v,-1,axis=1) - v) / grid.dy + \
          (np.roll(w,-1,axis=2) - w) / grid.dz
    
    phat = np.fft.fftn(grid.p)

    for index, value in np.ndenumerate(u):
        kk = grid.kxx[index[0]] + grid.kyy[index[1]] + grid.kzz[index[2]]
        if (not np.array_equal(index, [1,1,1])):
            phat /= kk
    
    p = np.fft.ifftn(phat).real

    u -= (p - np.roll(u,1,axis=0)) / grid.dx
    v -= (p - np.roll(v,1,axis=1)) / grid.dy
    w -= (p - np.roll(w,1,axis=2)) / grid.dz
    
    return u,v,w