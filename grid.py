import numpy as np

# Define periodic grid
class grid:
    
    # initialize grid of ones
    def __init__(self, Nx, Ny, Nz, dx, dy, dz):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx = dx
        self.dy = dy
        self.dz = dz

        # coordinates defined on the center
        # Indexed as u[X][Y][Z]
        x = np.reshape(np.linspace(dx/2,Nx*dx,Nx), (Nx, 1, 1))
        y = np.reshape(np.linspace(dy/2,Ny*dy,Ny), (1, Ny, 1))
        z = np.reshape(np.linspace(dz/2,Nz*dz,Nz), (1, 1, Nz))

        # duplicating row vectors to make a numpy array
        self.x = np.tile(x, (1,Ny,Nz))
        self.y = np.tile(y, (Nx,1,Nz))
        self.z = np.tile(z, (Nx,Ny,1))

        # initializing velocity, pressure fields with ones
        self.u = np.ones([Nx,Ny,Nz])
        self.v = np.ones([Nx,Ny,Nz])
        self.w = np.ones([Nx,Ny,Nz])
        self.p = np.ones([Nx,Ny,Nz])

    # read from file
    def read(self,filename):
        pass

    # Taylor-Green Vortex
    def vortex(self,A,B,C,a,b,c):
        if (A*a + B*b + C*c != 0):
            raise ValueError('Not well defined. A*a + B*b + C*C != 0.')
        
        # place u,v,w nodes in staggered grid
        self.u = A * np.cos(a*(self.x - self.dx)) * np.sin(b*self.y) * np.sin(c*self.z)
        self.v = B * np.sin(a*self.x) * np.cos(b*(self.y - self.dy)) * np.sin(c*self.z)
        self.w = C * np.sin(a*self.x) * np.sin(b*self.y) * np.cos(c*(self.z - self.dz))