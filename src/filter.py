import numpy as np
from src.grid import *
from src.compute_projection_step import *

def filter_grid(grid_DNS, grid_filtered):
    vx = np.append(range(0,int(grid_filtered.Nx/2)), range(grid_DNS.Nx - int(grid_filtered.Nx/2), grid_DNS.Nx))
    vy = np.append(range(0,int(grid_filtered.Ny/2)), range(grid_DNS.Ny - int(grid_filtered.Ny/2), grid_DNS.Ny))
    vz = np.append(range(0,int(grid_filtered.Nz/2)), range(grid_DNS.Nz - int(grid_filtered.Nz/2), grid_DNS.Nz))

    index = np.column_stack((vx,vy,vz))

    uhat = np.fft.fftn(grid_DNS.u)
    uhat_f = uhat[index] * (grid_filtered.Nx * grid_filtered.Ny * grid_filtered.Nz) / (grid_DNS.Nx * grid_DNS.Ny * grid_DNS.Nz)
    grid_filtered.u = np.fft.ifftn(uhat_f).real

    vhat = np.fft.fftn(grid_DNS.v)
    vhat_f = vhat[index] * (grid_filtered.Nx * grid_filtered.Ny * grid_filtered.Nz) / (grid_DNS.Nx * grid_DNS.Ny * grid_DNS.Nz)
    grid_filtered.v = np.fft.ifftn(vhat_f).real

    what = np.fft.fftn(grid_DNS.w)
    what_f = what[index] * (grid_filtered.Nx * grid_filtered.Ny * grid_filtered.Nz) / (grid_DNS.Nx * grid_DNS.Ny * grid_DNS.Nz)
    grid_filtered.w = np.fft.ifftn(what_f).real

    grid_filtered.define_wavenumber()

    grid_filtered.u, grid_filtered.v, grid_filtered.w =  compute_projection_step(grid_filtered)

    return grid_filtered