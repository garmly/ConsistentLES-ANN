import numpy as np
from src.filter import SGS_tensor
from src.grid import *
from src.roll_view import *

def compute_RS(grid):
    # compute strain rate tensor S_ij = 0.5 * (du_i / dx_j + du_j / dx_i)
    S_uu = (grid.u - roll_view(grid.u,-1,axis=0)) / grid.dx
    S_uv = 0.5*((grid.u - roll_view(grid.u,-1,axis=1)) / grid.dy + (grid.v - roll_view(grid.v,-1,axis=0)) / grid.dx)
    S_uw = 0.5*((grid.u - roll_view(grid.u,-1,axis=2)) / grid.dz + (grid.w - roll_view(grid.w,-1,axis=0)) / grid.dx)
    S_vu = S_uv
    S_vv = (grid.v - roll_view(grid.v,-1,axis=1)) / grid.dy
    S_vw = 0.5*((grid.v - roll_view(grid.v,-1,axis=2)) / grid.dz + (grid.w - roll_view(grid.w,-1,axis=1)) / grid.dy)
    S_wu = S_uw
    S_wv = S_vw
    S_ww = (grid.w - roll_view(grid.w,-1,axis=2)) / grid.dz

    # compute rotation rate tensor R_ij = 0.5 * (du_i / dx_j - du_j / dx_i)
    R_uu = 0 * grid.u
    R_uv = 0.5*((grid.u - roll_view(grid.u,-1,axis=1)) / grid.dy - (grid.v - roll_view(grid.v,-1,axis=0)) / grid.dx)
    R_uw = 0.5*((grid.u - roll_view(grid.u,-1,axis=2)) / grid.dz - (grid.w - roll_view(grid.w,-1,axis=0)) / grid.dx)
    R_vu = 0.5*((grid.v - roll_view(grid.v,-1,axis=0)) / grid.dx - (grid.u - roll_view(grid.u,-1,axis=1)) / grid.dy)
    R_vv = 0 * grid.v
    R_vw = 0.5*((grid.v - roll_view(grid.v,-1,axis=2)) / grid.dz - (grid.w - roll_view(grid.w,-1,axis=1)) / grid.dy)
    R_wu = 0.5*((grid.w - roll_view(grid.w,-1,axis=0)) / grid.dx - (grid.u - roll_view(grid.u,-1,axis=2)) / grid.dz)
    R_wv = 0.5*((grid.w - roll_view(grid.w,-1,axis=1)) / grid.dy - (grid.v - roll_view(grid.v,-1,axis=2)) / grid.dz)
    R_ww = 0 * grid.w

    R = SGS_tensor(R_uu,R_uv,R_uw,R_vu,R_vv,R_vw,R_wu,R_wv,R_ww)
    S = SGS_tensor(S_uu,S_uv,S_uw,S_vu,S_vv,S_vw,S_wu,S_wv,S_ww)

    return R,S