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

    u0, v0, w0 = grid.u, grid.v, grid.w
    u,v,w = u0,v0,w0

    Fu, Fv, Fw = np.zeros(3)
    for i in range(3):
        grid.u = u0 + h*np.dot(a[i:(i+1)])
        grid.v = v0 + h*np.dot(a[i:(i+1)])
        grid.w = w0 + h*np.dot(a[i:(i+1)])

        [Fu[i], Fv[i], Fw[i]] = compute_RHS(grid)
        [grid.u, grid.v, grid.w] = compute_projection_step(Fu[i], Fv[i], Fw[i])
    
    grid.u = u0 + h * np.dot(b,Fu)
    grid.u = v0 + h * np.dot(b,Fv)
    grid.u = w0 + h * np.dot(b,Fw)

    return grid