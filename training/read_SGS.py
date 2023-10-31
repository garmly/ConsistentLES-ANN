import os
import numpy as np
import torch

def read_SGS_binary(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")
    
    with open(filename, 'rb') as binfile:
        Nx = np.fromfile(binfile, dtype='>i4', count=1)[0]
        x = np.fromfile(binfile, dtype='>f8', count=Nx)
        Ny = np.fromfile(binfile, dtype='>i4', count=1)[0]
        y = np.fromfile(binfile, dtype='>f8', count=Ny)
        Nz = np.fromfile(binfile, dtype='>i4', count=1)[0]
        z = np.fromfile(binfile, dtype='>f8', count=Nz)

        data = np.zeros((Nx, Ny, Nz, 3, 3, 3), dtype='f8')
        for isubstep in range(3):
            NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
            uu = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
            NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
            uv = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
            NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
            uw = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
            NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
            vu = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
            NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
            vv = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
            NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
            vw = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
            NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
            wu = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
            NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
            wv = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
            NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
            ww = np.fromfile(binfile, dtype='>f8', count=Nx*Ny*Nz).reshape(Nx, Ny, Nz)
            
            data[:, :, :, 0, 0, isubstep] = uu
            data[:, :, :, 0, 1, isubstep] = uv
            data[:, :, :, 0, 2, isubstep] = uw
            data[:, :, :, 1, 0, isubstep] = vu
            data[:, :, :, 1, 1, isubstep] = vv
            data[:, :, :, 1, 2, isubstep] = vw
            data[:, :, :, 2, 0, isubstep] = wu
            data[:, :, :, 2, 1, isubstep] = wv
            data[:, :, :, 2, 2, isubstep] = ww

    return data

def read_delta(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")

    with open(filename, 'rb') as binfile:
        # Read grid dimensions
        Nx = np.fromfile(binfile, dtype='>i4', count=1)[0]
        x = np.fromfile(binfile, dtype='>f8', count=Nx)
        Ny = np.fromfile(binfile, dtype='>i4', count=1)[0]
        y = np.fromfile(binfile, dtype='>f8', count=Ny)
        Nz = np.fromfile(binfile, dtype='>i4', count=1)[0]
        z = np.fromfile(binfile, dtype='>f8', count=Nz)

        # Read delta components [ix,iy,iz,isubstep,Fu_i]
        delta = np.empty((Nx, Ny, Nz, 3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                NxNyNz = np.fromfile(binfile, dtype='>i4', count=3)
                delta[:, :, :, i, j] = np.fromfile(binfile, dtype='>f8', count=Nx * Ny * Nz).reshape(Nx, Ny, Nz)

    return delta
