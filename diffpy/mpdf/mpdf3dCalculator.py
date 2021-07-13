import numpy as np
from scipy import signal as sig
from scipy.spatial import KDTree as KDT

def gauss(grid,s=0.5):
    '''
    A function to generate a gaussian kernel of arbitrary size and density
    '''
    g = lambda point: 1/(s*np.sqrt((2*np.pi*s)**3))*np.exp(-1/2*(np.linalg.norm(point)/s)**2)
    return np.apply_along_axis(g,3,grid)

def vec_ac(a1,a2,delta):
    '''
    A function to implement the autocorrelation for two vector fields
    '''
    ac = sig.correlate(a1[:,:,:,0],a2[:,:,:,0],mode="same")*delta**3
    ac += sig.correlate(a1[:,:,:,1],a2[:,:,:,1],mode="same")*delta**3
    ac += sig.correlate(a1[:,:,:,2],a2[:,:,:,2],mode="same")*delta**3
    return ac

def vec_con(a1,a2,delta):
    '''
    A function to implement the convolution operator for two discrete vector fields
    ''' 
    con = sig.convolve(a1[:,:,:,0],a2[:,:,:,0],mode="same")*delta**3
    con += sig.convolve(a1[:,:,:,1],a2[:,:,:,1],mode="same")*delta**3
    con += sig.convolve(a1[:,:,:,2],a2[:,:,:,2],mode="same")*delta**3
    return con

def ups(grid):
    '''
    A function to generat the Upsilon filter from Roth et.al.
    '''
    g = lambda point: 0 if np.linalg.norm(point) == 0 else  point/np.linalg.norm(point)**4
    return np.apply_along_axis(g,3,grid)

class MPDF3Dcalculator:

    def __init__(self, magstruc=None, gaussPeakWidth=0.5, label=""):
        '''
        Need to decide on cononical input format
        '''
        if magstruc is None:
            self.magstruc = []
        else:
            self.magstruc = magstruc

        self.gaussPeakWidht = gaussPeakWidth
        self.label = label
        self.rgrid = None
        self.Nx = None
        self.Ny = None
        self.Nz = None
        self.dr = None

    def __repr__(self):
        if self.label == None:
            return "3dMPDFcalculator() object"
        else:
            return self.label +  ": 3dMPDFcalculator() object"

    def calc(self, verbose=False):
        '''
        Calculate and store the 3DMPDF 
        '''
        print("Running calc")
        self._make_rgrid()

        s_arr = np.zeros((self.Nx,self.Ny,self.Nz,3))
        if verbose :
            print("Setting up point spins")

        for i in range(len(self.magstruc.atoms)):
            idx = np.rint((self.magstruc.atoms[i] - self.rmin)/self.dr).astype(int) 

            if verbose:
                print(idx)
                print(self.magstruc.spins[i])

            s_arr[idx[0],idx[1],idx[2]] = self.magstruc.spins[i]

        s_3 = np.copy(s_arr)
        if verbose:
            print("Setting up filter grid")

        filter_x = np.arange(-3,3+self.dr,self.dr)
        X,Y,Z = np.meshgrid(filter_x,filter_x,filter_x,indexing='ij')
        filter_grid = np.moveaxis([X,Y,Z],0,-1)

        if verbose:
            print("Making filters")

        gaussian = gauss(filter_grid)
        upsilon = ups(filter_grid)

        if verbose:
            print("Convolving spin array")

        s_arr[:,:,:,0] = sig.convolve(s_arr[:,:,:,0],gaussian,mode='same')*self.dr**3
        s_arr[:,:,:,1] = sig.convolve(s_arr[:,:,:,1],gaussian,mode='same')*self.dr**3
        s_arr[:,:,:,2] = sig.convolve(s_arr[:,:,:,2],gaussian,mode='same')*self.dr**3

        if verbose:
            print("Computing mpdf")

        mag_ups = vec_con(s_arr,upsilon,self.dr)
        self.mpdf = vec_ac(s_arr,s_arr,self.dr) - 1/(np.pi**4)*sig.correlate(mag_ups,mag_ups,mode='same')*self.dr**3
        return s_arr,s_3

    def _make_rgrid(self, dr = 0.1):
        self.dr = dr
        pos = np.array([a for a in self.magstruc.atoms])
        print(type(pos))
        x_min = np.min(pos[:,0])
        x_max = np.max(pos[:,0])
        y_min = np.min(pos[:,1])
        y_max = np.max(pos[:,1])
        z_min = np.min(pos[:,2])
        z_max = np.max(pos[:,2])

        print("Min/Max defined")
        x = np.arange(x_min,x_max + dr, dr)
        y = np.arange(y_min,y_max + dr, dr)
        z = np.arange(z_min,z_max + dr, dr)
        N_x = len(x)
        N_y = len(y)
        N_z = len(z)
        print(N_x)
        print(N_y)
        print(N_z)
        print("Making meshgrid")
        X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
        print("Making rgrid")
        rgrid = np.moveaxis([X,Y,Z],0,-1)
        print("Saving sizes")
        self.Nx = N_x
        self.Ny = N_y
        self.Nz = N_z
        self.rmin = np.array([x_min,y_min,z_min])
        self.rmax = self.rmin + [(self.Nx-1)*self.dr,(self.Ny-1)*self.dr,(self.Nz-1)*self.dr]
        return rgrid

    def plot(self):
        '''
        Plot the 3DMPDF or spin configuration
        '''
        raise NotImplementedError()

    def run_checks(self):
        '''
        runs bounds and compatibility checks for internal variables. This should be called during __init__
        '''        
        raise NotImplementedError()

    def rgrid(self):
        '''
        Returns the real space grid points being used
        '''
        if self.dr is None:
            self.make_rgrid()
        X = np.np.arange(self.rmin[0],stop = self.rmax[0]+self.dr,step = self.dr)
        Y = np.np.arange(self.rmin[1],stop = self.rmax[1]+self.dr,step = self.dr)
        Z = np.np.arange(self.rmin[2],stop = self.rmax[2]+self.dr,step = self.dr)

        return X,Y,Z


    def copy(self):
        '''
        Return a deep copy of the object
        '''
        raise NotImplementedError()
