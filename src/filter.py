import copy as cp
import numpy as np
from src.grid import *
from src.compute_RHS import *
from src.compute_projection_step import *
from src.roll_view import *

def filter_grid(grid_DNS, grid_filtered):
    vx = np.expand_dims(np.append(range(0,grid_filtered.Nx//2), range(grid_DNS.Nx - grid_filtered.Nx//2, grid_DNS.Nx)),axis=(1,2))
    vy = np.expand_dims(np.append(range(0,grid_filtered.Ny//2), range(grid_DNS.Ny - grid_filtered.Ny//2, grid_DNS.Ny)),axis=(0,2))
    vz = np.expand_dims(np.append(range(0,grid_filtered.Nz//2), range(grid_DNS.Nz - grid_filtered.Nz//2, grid_DNS.Nz)),axis=(0,1))

    scaling = (grid_filtered.Nx * grid_filtered.Ny * grid_filtered.Nz) / (grid_DNS.Nx * grid_DNS.Ny * grid_DNS.Nz)

    def filter(field):
        fieldhat = np.fft.fftn(field)
        fieldhat_f = fieldhat[vx,vy,vz] * scaling
        return np.fft.ifftn(fieldhat_f).real

    # Calculating filtered u,v,w
    grid_filtered.u = filter(grid_DNS.u)
    grid_filtered.v = filter(grid_DNS.v)
    grid_filtered.w = filter(grid_DNS.w)

    # Calculating filtered Fu,Fv,Fw
    grid_filtered.Fu = filter(grid_DNS.Fu)
    grid_filtered.Fv = filter(grid_DNS.Fv)
    grid_filtered.Fw = filter(grid_DNS.Fw)

    # Removing divergence from velocity field
    grid_filtered.define_wavenumber()
    grid_filtered.u, grid_filtered.v, grid_filtered.w = compute_projection_step(grid_filtered)

    # Pre-computing terms
    u_roll_back = roll_view(grid_DNS.u,-1,axis=0)
    v_roll_back = roll_view(grid_DNS.v,-1,axis=1)
    w_roll_back = roll_view(grid_DNS.w,-1,axis=2)
    u_roll_v = roll_view(grid_DNS.u,1,axis=1)
    u_roll_w = roll_view(grid_DNS.u,1,axis=2)
    v_roll_u = roll_view(grid_DNS.v,1,axis=0)
    v_roll_w = roll_view(grid_DNS.v,1,axis=2)
    w_roll_u = roll_view(grid_DNS.w,1,axis=0)
    w_roll_v = roll_view(grid_DNS.w,1,axis=1)

    # Calculating remaining filtered SGS terms
    uu = (0.5*(grid_DNS.u + u_roll_back))**2
    uv = 0.5*(grid_DNS.u + u_roll_v) * 0.5*(grid_DNS.v + v_roll_u)
    uw = 0.5*(grid_DNS.u + u_roll_w) * 0.5*(grid_DNS.w + w_roll_u)
    vu = 0.5*(grid_DNS.v + v_roll_u) * 0.5*(grid_DNS.u + u_roll_v)
    vv = (0.5*(grid_DNS.v + v_roll_back))**2
    vw = 0.5*(grid_DNS.v + v_roll_w) * 0.5*(grid_DNS.w + w_roll_v)
    wu = 0.5*(grid_DNS.w + w_roll_u) * 0.5*(grid_DNS.u + u_roll_w)
    wv = 0.5*(grid_DNS.w + w_roll_v) * 0.5*(grid_DNS.v + v_roll_w)
    ww = (0.5*(grid_DNS.w + w_roll_back))**2

    uu_f = filter(uu)
    uv_f = filter(uv)
    uw_f = filter(uw)
    vu_f = filter(vu)
    vv_f = filter(vv)
    vw_f = filter(vw)
    wu_f = filter(wu)
    wv_f = filter(wv)
    ww_f = filter(ww)

    # Resolving SGS stress tensor
    u_roll_back_f = roll_view(grid_filtered.u,-1,axis=0)
    v_roll_back_f = roll_view(grid_filtered.v,-1,axis=1)
    w_roll_back_f = roll_view(grid_filtered.w,-1,axis=2)
    u_roll_v_f = roll_view(grid_filtered.u,1,axis=1)
    u_roll_w_f = roll_view(grid_filtered.u,1,axis=2)
    v_roll_u_f = roll_view(grid_filtered.v,1,axis=0)
    v_roll_w_f = roll_view(grid_filtered.v,1,axis=2)
    w_roll_u_f = roll_view(grid_filtered.w,1,axis=0)
    w_roll_v_f = roll_view(grid_filtered.w,1,axis=1)

    SGS_uu = uu_f - (0.5*(grid_filtered.u + u_roll_back_f))**2
    SGS_uv = uv_f - 0.5*(grid_filtered.u + u_roll_v_f) * 0.5*(grid_filtered.v + v_roll_u_f)
    SGS_uw = uw_f - 0.5*(grid_filtered.u + u_roll_w_f) * 0.5*(grid_filtered.w + w_roll_u_f)
    SGS_vu = vu_f - 0.5*(grid_filtered.v + v_roll_u_f) * 0.5*(grid_filtered.u + u_roll_v_f)
    SGS_vv = vv_f - (0.5*(grid_filtered.v + v_roll_back_f))**2
    SGS_vw = vw_f - 0.5*(grid_filtered.v + v_roll_w_f) * 0.5*(grid_filtered.w + w_roll_v_f)
    SGS_wu = wu_f - 0.5*(grid_filtered.w + w_roll_u_f) * 0.5*(grid_filtered.u + u_roll_w_f)
    SGS_wv = wv_f - 0.5*(grid_filtered.w + w_roll_v_f) * 0.5*(grid_filtered.v + v_roll_w_f)
    SGS_ww = ww_f - (0.5*(grid_filtered.w + w_roll_back_f))**2

    SGS = SGS_tensor(SGS_uu,SGS_uv,SGS_uw,SGS_vu,SGS_vv,SGS_vw,SGS_wu,SGS_wv,SGS_ww)

    return cp.deepcopy(grid_filtered), SGS

class SGS_tensor:
    
    # initialize grid of ones
    def __init__(self,uu,uv,uw,vu,vv,vw,wu,wv,ww):
        self.uu = uu
        self.uv = uv
        self.uw = uw
        self.vu = vu
        self.vv = vv
        self.vw = vw
        self.wu = wu
        self.wv = wv
        self.ww = ww