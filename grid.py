import numpy as np

# Define periodic grid
class grid:
    
    # initialize grid of ones
    def __init__(self, Nx, Ny, Nz, dx, dy, dz, nu):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nu = nu

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
        self.u = self.v = self.w = self.p = np.ones([Nx,Ny,Nz])

        # initializing time derivative fields
        self.Fu = self.Fv = self.Fw = np.ones([Nx,Ny,Nz])

        # initializing wavenumbers
        self.kxx = np.ones(Nx)
        self.kyy = np.ones(Ny)
        self.kzz = np.ones(Nz)

    # read from file
    def read(self,filename):
        pass

    def define_wavenumber(self):
        for i in range(self.Nx):
            self.kxx[i] = 2*(np.cos(2*np.pi*i/self.Nx) - 1)/self.dx**2 if i < self.Nx / 2 else \
                          2*(np.cos(2*np.pi*(-self.Nx+i)/self.Nx) - 1 )/self.dx**2
        
        for j in range(self.Ny):
            self.kyy[j] = 2*(np.cos(2*np.pi*j/self.Ny) - 1)/self.dy**2 if i < self.Ny / 2 else \
                          2*(np.cos(2*np.pi*(-self.Ny+j)/self.Ny) - 1 )/self.dy**2
        
        for k in range(self.Nz):
            self.kzz[i] = 2*(np.cos(2*np.pi*i/self.Nz) - 1)/self.dz**2 if i > self.Nz / 2 else \
                          2*(np.cos(2*np.pi*(-self.Nz+i)/self.Nz) - 1 )/self.dz**2

    # Taylor-Green Vortex
    def vortex(self,A,B,C,a,b,c):
        if (A*a + B*b + C*c != 0):
            raise ValueError('Not well defined. A*a + B*b + C*C != 0.')
        
        # place u,v,w nodes in staggered grid
        self.u = A * np.cos(a*(self.x - self.dx)) * np.sin(b*self.y) * np.sin(c*self.z)
        self.v = B * np.sin(a*self.x) * np.cos(b*(self.y - self.dy)) * np.sin(c*self.z)
        self.w = C * np.sin(a*self.x) * np.sin(b*self.y) * np.cos(c*(self.z - self.dz))