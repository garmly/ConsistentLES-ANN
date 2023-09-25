import csv
import numpy as np

def read_SGS(filename, Nx,Ny,Nz):
    # read in SGS stress tensor from file
    # returns SGS tensor as a numpy array
    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        reader.__next__()
        tau = np.zeros([Nx*Ny*Nz,9])
        for i,row in enumerate(reader):
            for j,val in enumerate(row):
                if j > 2 and i < Nx*Ny*Nz:
                    tau[i,j-3] = float(val)
    tau = tau.reshape([Nx,Ny,Nz,3,3])
    return tau
