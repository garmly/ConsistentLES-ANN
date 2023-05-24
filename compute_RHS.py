import numpy as np

# takes in u, v and computes the RHS of the Navier-Stokes
def compute_RHS(grid):

    # Computing Fu
    Fu = np.zeros(grid.u.shape)

    # d(uu)/dx
    uu = (0.5*(grid.u + np.roll(grid.u,-1,axis=0)))**2
    Fu -= (uu - np.roll(uu,1,axis=0)) / grid.dx

    # d(uv)/dy
    uv = 0.5*(grid.u + np.roll(grid.u,1,axis=1)) * \
         0.5*(grid.v + np.roll(grid.v,1,axis=0))
    Fu -= (np.roll(uv,-1,axis=1) - uv) / grid.dy

    # d(uw)/dz
    uw = 0.5*(grid.u + np.roll(grid.u,1,axis=2)) * \
         0.5*(grid.w + np.roll(grid.w,1,axis=0))
    Fu -= (np.roll(uw,-1,axis=1) - uw) / grid.dz

    # 1/Re * d^2(u)/d(x)^2
    Fu += grid.nu *(np.roll(grid.u,-1,axis=0) - 2*grid.u + np.roll(grid.u,1,axis=0)) / grid.dx**2
    # 1/Re * d^2(u)/d(y)^2
    Fu += grid.nu *(np.roll(grid.u,-1,axis=1) - 2*grid.u + np.roll(grid.u,1,axis=1)) / grid.dy**2
    # 1/Re * d^2(u)/d(z)^2
    Fu += grid.nu *(np.roll(grid.u,-1,axis=2) - 2*grid.u + np.roll(grid.u,1,axis=2)) / grid.dz**2

    # Computing Fv
    Fv = np.zeros(grid.v.shape)

    # d(uv)/dx
    uv = 0.5*(grid.v + np.roll(grid.v,1,axis=0)) * \
         0.5*(grid.u + np.roll(grid.u,1,axis=1))
    Fv -= (np.roll(uv,-1,axis=1) - uv) / grid.dx

    # d(vv)/dy
    vv = (0.5*(grid.v + np.roll(grid.v,-1,axis=1)))**2
    Fv -= (vv - np.roll(vv,1,axis=0)) / grid.dy

    # d(vw)/dz
    vw = 0.5*(grid.v + np.roll(grid.v,1,axis=2)) * \
         0.5*(grid.w + np.roll(grid.w,1,axis=1))
    Fv -= (np.roll(vw,-1,axis=1) - vw) / grid.dz

    # 1/Re * d^2(v)/d(x)^2
    Fv += grid.nu *(np.roll(grid.v,-1,axis=0) - 2*grid.v + np.roll(grid.v,1,axis=0)) / grid.dx**2
    # 1/Re * d^2(v)/d(y)^2
    Fv += grid.nu *(np.roll(grid.v,-1,axis=1) - 2*grid.v + np.roll(grid.v,1,axis=1)) / grid.dy**2
    # 1/Re * d^2(v)/d(z)^2
    Fv += grid.nu *(np.roll(grid.v,-1,axis=2) - 2*grid.v + np.roll(grid.v,1,axis=2)) / grid.dz**2

    # Computing Fw
    Fw = np.zeros(grid.w.shape)

    # d(uw)/dx
    uw = 0.5*(grid.w + np.roll(grid.w,1,axis=0)) * \
         0.5*(grid.u + np.roll(grid.u,1,axis=2))
    Fw -= (np.roll(uw,-1,axis=1) - uw) / grid.dx

    # d(uw)/dx
    vw = 0.5*(grid.w + np.roll(grid.w,1,axis=1)) * \
         0.5*(grid.v + np.roll(grid.v,1,axis=2))
    Fw -= (np.roll(vw,-1,axis=1) - vw) / grid.dy

    # d(ww)/dz
    ww = (0.5*(grid.w + np.roll(grid.w,-1,axis=1)))**2
    Fw -= (ww - np.roll(ww,1,axis=0)) / grid.dz

    return Fu, Fv, Fw