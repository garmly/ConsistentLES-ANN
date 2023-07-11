import numpy as np

# takes in u, v and computes the RHS of the Navier-Stokes
def compute_RHS(grid, LES, forcing=1, SGS=None):
        
    # Pre-computing terms
    u_roll_back = np.roll(grid.u,-1,axis=0)
    v_roll_back = np.roll(grid.v,-1,axis=1)
    w_roll_back = np.roll(grid.w,-1,axis=2)
    u_roll_v = np.roll(grid.u,1,axis=1)
    u_roll_w = np.roll(grid.u,1,axis=2)
    v_roll_u = np.roll(grid.v,1,axis=0)
    v_roll_w = np.roll(grid.v,1,axis=2)
    w_roll_u = np.roll(grid.w,1,axis=0)
    w_roll_v = np.roll(grid.w,1,axis=1)

    # Computing Fu
    Fu = np.zeros(grid.u.shape)

    # d(uu)/dx
    uu = (0.5*(grid.u + u_roll_back))**2
    Fu -= (uu - np.roll(uu,1,axis=0)) / grid.dx

    # d(uv)/dy
    uv = 0.5*(grid.u + u_roll_v) * \
         0.5*(grid.v + v_roll_u)
    Fu -= (np.roll(uv,-1,axis=1) - uv) / grid.dy

    # d(uw)/dz
    uw = 0.5*(grid.u + u_roll_w) * \
         0.5*(grid.w + w_roll_u)
    Fu -= (np.roll(uw,-1,axis=2) - uw) / grid.dz

    # 1/Re * d^2(u)/d(x)^2
    Fu += grid.nu *(u_roll_back - 2*grid.u + np.roll(grid.u,1,axis=0)) / grid.dx**2
    # 1/Re * d^2(u)/d(y)^2
    Fu += grid.nu *(np.roll(grid.u,-1,axis=1) - 2*grid.u + u_roll_v) / grid.dy**2
    # 1/Re * d^2(u)/d(z)^2
    Fu += grid.nu *(np.roll(grid.u,-1,axis=2) - 2*grid.u + u_roll_w) / grid.dz**2

    # Computing Fv
    Fv = np.zeros(grid.v.shape)

    # d(uv)/dx
    uv = 0.5*(grid.v + v_roll_u) * \
         0.5*(grid.u + u_roll_v)
    Fv -= (np.roll(uv,-1,axis=0) - uv) / grid.dx

    # d(vv)/dy
    vv = (0.5*(grid.v + v_roll_back))**2
    Fv -= (vv - np.roll(vv,1,axis=1)) / grid.dy

    # d(vw)/dz
    vw = 0.5*(grid.v + v_roll_w) * \
         0.5*(grid.w + w_roll_v)
    Fv -= (np.roll(vw,-1,axis=2) - vw) / grid.dz

    # 1/Re * d^2(v)/d(x)^2
    Fv += grid.nu *(np.roll(grid.v,-1,axis=0) - 2*grid.v + v_roll_u) / grid.dx**2
    # 1/Re * d^2(v)/d(y)^2
    Fv += grid.nu *(v_roll_back - 2*grid.v + np.roll(grid.v,1,axis=1)) / grid.dy**2
    # 1/Re * d^2(v)/d(z)^2
    Fv += grid.nu *(np.roll(grid.v,-1,axis=2) - 2*grid.v + v_roll_w) / grid.dz**2

    # Computing Fw
    Fw = np.zeros(grid.w.shape)

    # d(uw)/dx
    uw = 0.5*(grid.w + w_roll_u) * \
         0.5*(grid.u + u_roll_w)
    Fw -= (np.roll(uw,-1,axis=0) - uw) / grid.dx

    # d(uw)/dx
    vw = 0.5*(grid.w + w_roll_v) * \
         0.5*(grid.v + v_roll_w)
    Fw -= (np.roll(vw,-1,axis=1) - vw) / grid.dy

    # d(ww)/dz
    ww = (0.5*(grid.w + w_roll_back))**2
    Fw -= (ww - np.roll(ww,1,axis=2)) / grid.dz

    # 1/Re * d^2(w)/d(x)^2
    Fw += grid.nu *(np.roll(grid.w,-1,axis=0) - 2*grid.w + w_roll_u) / grid.dx**2
    # 1/Re * d^2(w)/d(y)^2
    Fw += grid.nu *(np.roll(grid.w,-1,axis=1) - 2*grid.w + w_roll_v) / grid.dy**2
    # 1/Re * d^2(w)/d(z)^2
    Fw += grid.nu *(w_roll_back - 2*grid.w + np.roll(grid.w,1,axis=2)) / grid.dz**2

    if LES:
          Fu -= (np.roll(SGS.uu,-1,axis=0) - SGS.uu) / grid.dx + \
                (np.roll(SGS.uv,-1,axis=0) - SGS.uv) / grid.dy + \
                (np.roll(SGS.uw,-1,axis=0) - SGS.uw) / grid.dz
          Fv -= (np.roll(SGS.vu,-1,axis=0) - SGS.vu) / grid.dx + \
                (np.roll(SGS.vv,-1,axis=0) - SGS.vv) / grid.dy + \
                (np.roll(SGS.vw,-1,axis=0) - SGS.vw) / grid.dz
          Fw -= (np.roll(SGS.wu,-1,axis=0) - SGS.wu) / grid.dx + \
                (np.roll(SGS.wv,-1,axis=0) - SGS.wv) / grid.dy + \
                (np.roll(SGS.ww,-1,axis=0) - SGS.ww) / grid.dz

    Fu += grid.u + forcing
    Fv += grid.v + forcing
    Fw += grid.w + forcing

    return Fu, Fv, Fw