import numpy as np
from grid import *
from compute_RHS import *
from compute_projection_step import *

def time_advance_RK3(grid):
    # weights, nodes, and Runge-Kutta matrix
    b = np.array([1/6, 2/3, 1/6])
    c = np.array([0, 0.5, 1])
    a = np.array([[0,0,0],[1/2,0,0],[-1,2,0]])

    # Courant Number 1 < 1.75
    C = 1.00

    # Maximum timestep for numerical stability (CFL condition)
    h = abs(min(C*grid.dx/np.max(grid.u), C*grid.dy/np.max(grid.v)))

    # initial conditions
    u0, v0, w0 = grid.u, grid.v, grid.w
    u,v,w = u0,v0,w0

    # Have d/dx_i F each be a numpy array
    Fu = np.zeros([grid.Nx,grid.Ny,grid.Nz,3])
    Fv = np.zeros([grid.Nx,grid.Ny,grid.Nz,3])
    Fw = np.zeros([grid.Nx,grid.Ny,grid.Nz,3])

    for i in range(3):
        grid.u = u0 + h*np.sum(Fu * a[i,:], axis=-1)
        grid.v = v0 + h*np.sum(Fv * a[i,:], axis=-1)
        grid.w = w0 + h*np.sum(Fw * a[i,:], axis=-1)

        # remove divergence and compute RHS of Navier-Stokes
        grid.u, grid.v, grid.w = compute_projection_step(grid,True)
        grid.Fu, grid.Fv, grid.Fw = compute_RHS(grid)

        # remove divergence from Fu, Fv, Fw
        Fu[..., i], Fv[..., i], Fw[..., i] = compute_projection_step(grid,False)

    grid.u = u0 + h * np.sum(Fu * b, axis=-1)
    grid.v = v0 + h * np.sum(Fv * b, axis=-1)
    grid.w = w0 + h * np.sum(Fw * b, axis=-1)

    return grid, h