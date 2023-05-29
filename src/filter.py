import numpy as np
from compute_projection_step import *

def filter_grid(grid_DNS, grid_filtered):
    vx = [np.linspace(0,int(grid_DNS.Nx/2)), np.linspace(grid_DNS.Nx - int(grid_filtered.Nx/2) + 1, grid_DNS.Nx)]
    vy = [np.linspace(0,int(grid_DNS.Ny/2)), np.linspace(grid_DNS.Ny - int(grid_filtered.Ny/2) + 1, grid_DNS.Ny)]
    vz = [np.linspace(0,int(grid_DNS.Nz/2)), np.linspace(grid_DNS.Nz - int(grid_filtered.Nz/2) + 1, grid_DNS.Nz)]

    uhat = np.fft.fftn(grid_DNS.u)
    uhat_f = uhat[vx,vy,vz] * (grid_filtered.Nx * grid_filtered.Ny * grid_filtered.Nz) * (grid_DNS.Nx * grid_DNS.Ny * grid_DNS.Nz)
    u_f = np.fft.ifftn(uhat_f).real

    vhat = np.fft.fftn(grid_DNS.v)
    vhat_f = uhat[vx,vy,vz] * (grid_filtered.Nx * grid_filtered.Ny * grid_filtered.Nz) * (grid_DNS.Nx * grid_DNS.Ny * grid_DNS.Nz)
    v_f = np.fft.ifftn(vhat_f).real

    what = np.fft.fftn(grid_DNS.w)
    what_f = what[vx,vy,vz] * (grid_filtered.Nx * grid_filtered.Ny * grid_filtered.Nz) * (grid_DNS.Nx * grid_DNS.Ny * grid_DNS.Nz)
    w_f = np.fft.ifftn(what_f).real

    return compute_projection_step(grid_filtered)