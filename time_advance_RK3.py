import numpy as np

def time_advance_RK3(grid):
    # weights, nodes, and Runge-Kutta matrix
    b = np.array([1/6, 2/3, 1/6])
    c = np.array([0, 0.5, 1])
    a = np.array([[0,0,0],[1/2,0,0],[-1,2,0]])

    # Courant Number 1 < 1.75
    C = 1.00

    # Maximum timestep for numerical stability (CFL condition)
    h = abs(min(C*grid.dx/np.max(grid.u), C*grid.dy/np.max(grid.v)))

    k = np.zeros(3)
    