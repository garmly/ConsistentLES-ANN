import csv
import copy as cp
from src.grid import *
from src.filter import *
from src.roll_view import *
from src.interface import *
from src.time_advance_RK3 import *

print("INITIALIZING...")

# reading from file if read is True
if read:
    grid_DNS = read_grid("in/"+filename, nu)
    dx = grid_DNS.dx
    dy = grid_DNS.dy
    dz = grid_DNS.dz
    Lx = (grid_DNS.x[-1,0,0] + dx) / (2 * np.pi)
    Ly = (grid_DNS.y[0,-1,0] + dy) / (2 * np.pi)
    Lz = (grid_DNS.z[0,0,-1] + dz) / (2 * np.pi)
    Nx = grid_DNS.Nx
    Ny = grid_DNS.Ny
    Nz = grid_DNS.Nz
else:
    dx = Lx * np.pi * 2 / Nx
    dy = Ly * np.pi * 2 / Ny
    dz = Lz * np.pi * 2 / Nz

# initializing simulation variables
time = 0
verbose = True
sample_index = [Nx//2,Ny//2,Nz//2]

# Defining filtered quantities
dxf = Lx * np.pi * 2 / Nxf
dyf = Ly * np.pi * 2 / Nyf
dzf = Lz * np.pi * 2 / Nzf

Re = U0 * Lx / nu

# initializing grids
grid_filter = grid(Nxf,Nyf,Nzf,dxf,dyf,dzf,nu)

if not read:
    grid_DNS = grid(Nx,Ny,Nz,dx,dy,dz,nu)
    grid_DNS.vortex(-4,1,2,1,2,1)

grid_DNS.define_wavenumber()

# Defining LES grid
grid_LES_corrected, SGS = filter_grid(grid_DNS, grid_filter)
grid_LES_uncorrected = cp.deepcopy(grid_LES_corrected)

print("==========================================")
print("READING: " + filename if read else "READING: N/A")
print("Nx: " + str(Nx))
print("Ny: " + str(Ny))
print("Nz: " + str(Nz))
print("Lx: " + str(Lx))
print("Ly: " + str(Ly))
print("Lz: " + str(Lz))
print("==========================================")

i = 1                   # iteration
tvals = np.array([])    # time values
uvals = np.array([])    # u values
vvals = np.array([])    # v values
wvals = np.array([])    # w values
rsdlsu = np.array([0])  # residual values of u
rsdlsv = np.array([0])  # residual values of v
rsdlsw = np.array([0])  # residual values of w

grid_DNS.u, grid_DNS.v, grid_DNS.w = compute_projection_step(grid_DNS)
grid_filter, SGS = filter_grid(grid_DNS, grid_filter)

while (time < max_time):
    grid_DNS, h = time_advance_RK3(grid_DNS, LES=False)
    grid_LES_uncorrected, h = time_advance_RK3(grid_LES_uncorrected, LES=True, timeControl=h, SGS_tensor=SGS)

    delta_u = grid_filter.u - grid_LES_corrected.u
    delta_v = grid_filter.v - grid_LES_corrected.v
    delta_w = grid_filter.w - grid_LES_corrected.w
    delta_p = grid_filter.p - grid_LES_corrected.p

    grid_LES_corrected, h = time_advance_RK3(grid_LES_corrected, LES=True, timeControl=h, SGS_tensor=SGS)
    
    grid_LES_corrected.u += delta_u
    grid_LES_corrected.v += delta_v
    grid_LES_corrected.w += delta_w
    grid_LES_corrected.p += delta_p

    grid_filter, SGS = filter_grid(grid_DNS, grid_filter)
    
    if (verbose):
        # get the point located at the middle of the grid
        uvals = np.append(uvals,grid_DNS.u[sample_index])
        vvals = np.append(vvals,grid_DNS.v[sample_index])
        wvals = np.append(wvals,grid_DNS.w[sample_index])
        tvals = np.append(tvals,time)

        rsdlsu = np.append(rsdlsu, abs(uvals[i] - uvals[i-1]))
        rsdlsv = np.append(rsdlsv, abs(vvals[i] - vvals[i-1]))
        rsdlsw = np.append(rsdlsw, abs(vvals[i] - vvals[i-1]))

        print("TIME: " + str(time))
        print("==========================================")
        print("dT: " + str(h))
        print("UVAL: " + str(uvals[i]))
        print("VVAL: " + str(vvals[i]))
        print("WVAL: " + str(vvals[i]))
        print("URSD: " + str(abs(uvals[i] - uvals[i-1])))
        print("VRSD: " + str(abs(vvals[i] - vvals[i-1])))
        print("WRSD: " + str(abs(wvals[i] - wvals[i-1])))
        print("==========================================")

    # check for divergence-free velocity field
    div = (roll_view(grid_DNS.u,-1,axis=0) - grid_DNS.u) / grid_DNS.dx + \
          (roll_view(grid_DNS.v,-1,axis=1) - grid_DNS.v) / grid_DNS.dy + \
          (roll_view(grid_DNS.w,-1,axis=2) - grid_DNS.w) / grid_DNS.dz
    
    if np.max(np.abs(div)) > 1e-10:
        raise ValueError('Velocity field is not divergence free. Max(div) = ' + str(np.max(div)))
    
    if (i % write_interval == 0):
        with open('./out/unfiltered/t' + str(i) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["x","y","z","u","v","w","p"])
            data = np.column_stack((grid_DNS.x.flatten(),
                                    grid_DNS.y.flatten(),
                                    grid_DNS.z.flatten(),
                                    grid_DNS.u.flatten(),
                                    grid_DNS.v.flatten(),
                                    grid_DNS.w.flatten(),
                                    grid_DNS.p.flatten()))
            writer.writerows(data)

        with open('./out/filtered/raw/t' + str(i) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["x","y","z","uf","vf","wf","pf"])
            data = np.column_stack((grid_filter.x.flatten(),
                                    grid_filter.y.flatten(),
                                    grid_filter.z.flatten(),
                                    grid_filter.u.flatten(),
                                    grid_filter.v.flatten(),
                                    grid_filter.w.flatten(),
                                    grid_filter.p.flatten()))
            writer.writerows(data)

        with open('./out/filtered/SGS/t' + str(i) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["x","y","z","uu","uv","uw","vu","vv","vw","wu","wv","ww"])
            data = np.column_stack((grid_filter.x.flatten(),
                                    grid_filter.y.flatten(),
                                    grid_filter.z.flatten(),
                                    SGS.uu.flatten(),
                                    SGS.uv.flatten(),
                                    SGS.uw.flatten(),
                                    SGS.vu.flatten(),
                                    SGS.vv.flatten(),
                                    SGS.vw.flatten(),
                                    SGS.wu.flatten(),
                                    SGS.wv.flatten(),
                                    SGS.ww.flatten()))
            writer.writerows(data)

        with open('./out/filtered/LES_corrected/t' + str(i) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["x","y","z","u_LES_cor","v_LES_cor","w_LES_cor","p_LES_cor"])
            data = np.column_stack((grid_LES_corrected.x.flatten(),
                                    grid_LES_corrected.y.flatten(),
                                    grid_LES_corrected.z.flatten(),
                                    grid_LES_corrected.u.flatten(),
                                    grid_LES_corrected.v.flatten(),
                                    grid_LES_corrected.w.flatten(),
                                    grid_LES_corrected.p.flatten()))
            writer.writerows(data)

        with open('./out/filtered/LES_uncorrected/t' + str(i) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["x","y","z","u_LES_ucor","v_LES_ucor","w_LES_ucor","p_LES_ucor"])
            data = np.column_stack((grid_LES_uncorrected.x.flatten(),
                                    grid_LES_uncorrected.y.flatten(),
                                    grid_LES_uncorrected.z.flatten(),
                                    grid_LES_uncorrected.u.flatten(),
                                    grid_LES_uncorrected.v.flatten(),
                                    grid_LES_uncorrected.w.flatten(),
                                    grid_LES_uncorrected.p.flatten()))
            writer.writerows(data)

        with open('./out/filtered/delta/corrected/t' + str(i) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["x","y","z","delta_u","delta_v","delta_w","delta_p"])
            data = np.column_stack((grid_LES_corrected.x.flatten(),
                                    grid_LES_corrected.y.flatten(),
                                    grid_LES_corrected.z.flatten(),
                                    delta_u.flatten(),
                                    delta_v.flatten(),
                                    delta_w.flatten(),
                                    delta_p.flatten()))
            writer.writerows(data)

        with open('./out/filtered/delta/uncorrected/t' + str(i) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["x","y","z","delta_u","delta_v","delta_w","delta_p"])
            data = np.column_stack((grid_LES_uncorrected.x.flatten(),
                                    grid_LES_uncorrected.y.flatten(),
                                    grid_LES_corrected.z.flatten(),
                                    grid_filter.u.flatten() - grid_LES_uncorrected.u.flatten(),
                                    grid_filter.v.flatten() - grid_LES_uncorrected.v.flatten(),
                                    grid_filter.w.flatten() - grid_LES_uncorrected.w.flatten(),
                                    grid_filter.p.flatten() - grid_LES_uncorrected.p.flatten()))
            writer.writerows(data)

    i += 1
    time += h
