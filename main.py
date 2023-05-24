from grid import *
from time_advance_RK3 import *
import matplotlib.pyplot as plt

# TODO: Replace with user input
Lx = 1
Ly = 1
Lz = 1
Nx = 32
Ny = 32
Nz = 32
dx = np.pi * 2 / Nx
dy = np.pi * 2 / Ny
dz = np.pi * 2 / Nz
time = 0
U0 = 1
nu = 1e-3
Re = U0 * Lx / nu
verbose = True

# initializing grid
grid_DNS = grid(Nx,Ny,Nz,dx,dy,dz,nu)
grid_DNS.vortex(-5,2,2.5,2,2.5,2)
grid_DNS.define_wavenumber()

i = 0                   # iteration
tvals = np.array([])    # time values
uvals = np.array([])    # u values
vvals = np.array([])    # v values
wvals = np.array([])    # w values
rsdlsu = np.array([0])  # residual values of u
rsdlsv = np.array([0])  # residual values of v
rsdlsv = np.array([0])  # residual values of w

while (time < 3):
    
    if (i % 100 == 0):
        plt.imshow(grid_DNS.u[...,8], interpolation='nearest')
        plt.show()

    grid_DNS, h = time_advance_RK3(grid_DNS)
    
    if (verbose):
        # get the point located at the middle of the grid
        uvals = np.append(uvals,grid_DNS.u[int(Nx/2)][int(Ny/2)][int(Nz/2)])
        vvals = np.append(vvals,grid_DNS.v[int(Nx/2)][int(Ny/2)][int(Nz/2)])
        wvals = np.append(wvals,grid_DNS.w[int(Nx/2)][int(Ny/2)][int(Nz/2)])
        tvals = np.append(tvals,time)

        if (i > 1):
            rsdlsu = np.append(rsdlsu, abs(uvals[i] - uvals[i-1]))
            rsdlsv = np.append(rsdlsv, abs(vvals[i] - vvals[i-1]))

            print("TIME: " + str(time))
            print("==========================================")
            print("dT: " + str(h))
            print("UVAL: " + str(uvals[i]))
            print("VVAL: " + str(vvals[i]))
            print("URSD: " + str(abs(uvals[i] - uvals[i-1])))
            print("VRSD: " + str(abs(vvals[i] - vvals[i-1])))
            print("==========================================")
    
    # check for divergence-free velocity field    
    if (np.max((np.roll(grid_DNS.u,-1,axis=0) - grid_DNS.u) / grid_DNS.dx + \
                (np.roll(grid_DNS.v,-1,axis=1) - grid_DNS.v) / grid_DNS.dy + \
                (np.roll(grid_DNS.w,-1,axis=2) - grid_DNS.w) / grid_DNS.dz) > 1e+10):
        raise ValueError('Velocity field is not divergence free')
        
    i += 1
    time += h

plt.plot(tvals,uvals)
plt.plot(tvals,vvals)
plt.plot(tvals,wvals)
plt.show()