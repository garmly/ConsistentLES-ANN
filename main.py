from grid import *

global nu,dx,dy,dz,dt,kxx,kyy,kzz,Nx,Ny,Nz,Re

# TODO: Replace with user input
Lx = 1
Ly = 1
Lz = 1
Nx = 64
Ny = 64
Nz = 64
dx = np.pi * 2 / Nx
dy = np.pi * 2 / Ny
dz = np.pi * 2 / Nz
time = 0
U0 = 1
Re = U0 * Lx / 1e-3

# initializing grid
grid = grid(Nx,Ny,Nz,dx,dy,dz)