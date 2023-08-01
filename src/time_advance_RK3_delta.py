import numpy as np
import copy as cp
from src.filter import *
from src.grid import *
from src.compute_RHS import *
from src.compute_projection_step import *

def time_advance_RK3_delta(grid_LES, grid_DNS, timeControl=None):
    """
    Computes corrected velocity field using RK3

    Inputs:
        grid_LES: LES grid object
        grid_DNS: DNS grid object
        timeControl: time step (optional)
    """

    # weights, nodes, and Runge-Kutta matrix
    b = np.array([1/6, 2/3, 1/6])
    c = np.array([0, 0.5, 1])
    a = np.array([[0,0,0],[1/2,0,0],[-1,2,0]])

    # Courant Number 1 < 1.75
    C = 1.00

    # Maximum timestep for numerical stability (CFL condition)
    if not timeControl:
        h = abs(min(C*grid_LES.dx/np.max(grid_LES.u), C*grid_LES.dy/np.max(grid_LES.v), C*grid_LES.dz/np.max(grid_LES.w)))
    else:
        h = timeControl

    # initial conditions
    u0_f, v0_f, w0_f = grid_LES.u, grid_LES.v, grid_LES.w
    u0_DNS, v0_DNS, w0_DNS = grid_DNS.u, grid_DNS.v, grid_DNS.w

    # Have dFx_i/dt each be a numpy array
    Fu_f = np.zeros([grid_LES.Nx,grid_LES.Ny,grid_LES.Nz,3])
    Fv_f = np.zeros([grid_LES.Nx,grid_LES.Ny,grid_LES.Nz,3])
    Fw_f = np.zeros([grid_LES.Nx,grid_LES.Ny,grid_LES.Nz,3])
    
    # Do the same for DNS
    Fu = np.zeros([grid_DNS.Nx,grid_DNS.Ny,grid_DNS.Nz,3])
    Fv = np.zeros([grid_DNS.Nx,grid_DNS.Ny,grid_DNS.Nz,3])
    Fw = np.zeros([grid_DNS.Nx,grid_DNS.Ny,grid_DNS.Nz,3])

    # define filtered grid and delta
    grid_filtered = cp.deepcopy(grid_LES)
    delta = np.zeros([grid_LES.Nx,grid_LES.Ny,grid_LES.Nz,3,3])

    for i in range(3):
        grid_LES.u = u0_f + h*np.sum(Fu_f * a[i,:], axis=-1)
        grid_LES.v = v0_f + h*np.sum(Fv_f * a[i,:], axis=-1)
        grid_LES.w = w0_f + h*np.sum(Fw_f * a[i,:], axis=-1)
        grid_DNS.u = u0_DNS + h*np.sum(Fu * a[i,:], axis=-1)
        grid_DNS.v = v0_DNS + h*np.sum(Fv * a[i,:], axis=-1)
        grid_DNS.w = w0_DNS + h*np.sum(Fw * a[i,:], axis=-1)

        # remove divergence and compute RHS of Navier-Stokes
        grid_DNS.u, grid_DNS.v, grid_DNS.w = compute_projection_step(grid_DNS,True)

        # compute RHS and delta
        compute_RHS(grid_DNS)
        grid_filtered, SGS_f = filter_grid(grid_DNS, grid_filtered)
        compute_RHS(grid_LES, SGS=SGS_f)

        delta[:,:,:,i,0] = grid_filtered.Fu - grid_LES.Fu
        delta[:,:,:,i,1] = grid_filtered.Fv - grid_LES.Fv
        delta[:,:,:,i,2] = grid_filtered.Fw - grid_LES.Fw
        
        grid_LES.Fu += delta[:,:,:,i,0]
        grid_LES.Fv += delta[:,:,:,i,1]
        grid_LES.Fw += delta[:,:,:,i,2]

        maxdiff = np.max([grid_LES.Fu - grid_filtered.Fu, \
                          grid_LES.Fv - grid_filtered.Fv, \
                          grid_LES.Fw - grid_filtered.Fw])
        
        if maxdiff > 1e-5:
            raise ValueError('Filtered RHS and LES RHS do not match. Max(Fu_f - Fu_LES) = ' + str(maxdiff) + '.)')

        # remove divergence from Fu, Fv, Fw
        Fu_f[:,:,:,i], Fv_f[:,:,:,i], Fw_f[:,:,:,i] = compute_projection_step(grid_LES,False)
        Fu[:,:,:,i], Fv[:,:,:,i], Fw[:,:,:,i] = compute_projection_step(grid_DNS,False)

    grid_LES.u = u0_f + h * np.sum(Fu_f * b, axis=-1)
    grid_LES.v = v0_f + h * np.sum(Fv_f * b, axis=-1)
    grid_LES.w = w0_f + h * np.sum(Fw_f * b, axis=-1)
    grid_DNS.u = u0_DNS + h * np.sum(Fu * b, axis=-1)
    grid_DNS.v = v0_DNS + h * np.sum(Fv * b, axis=-1)
    grid_DNS.w = w0_DNS + h * np.sum(Fw * b, axis=-1)

    #grid_LES.u, grid_LES.v, grid_LES.w = compute_projection_step(grid_LES,True)
    grid_DNS.u, grid_DNS.v, grid_DNS.w = compute_projection_step(grid_DNS,True)

    return grid_LES, h, delta