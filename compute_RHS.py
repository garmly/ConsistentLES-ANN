# takes in u, v and computes the RHS of the Navier-Stokes
def compute_RHS(grid):
    return grid.u, grid.v, grid.w