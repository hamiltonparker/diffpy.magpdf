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
    print("**1**")
    ac += sig.correlate(a1[:,:,:,1],a2[:,:,:,1],mode="same")*delta**3
    print("**2**")
    ac += sig.correlate(a1[:,:,:,2],a2[:,:,:,2],mode="same")*delta**3
    print("**3**")
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
        if verbose:
            print("comp1")
        comp1 = vec_ac(s_arr,s_arr,self.dr)
        if verbose:
            print("comp2")
        comp2 = sig.correlate(mag_ups,mag_ups,mode='same')*self.dr**3
        if verbose:
            print("mpdf")
        self.mpdf = comp1 - 1/(np.pi**4)*comp2
        return s_arr,comp1,comp2,mag_ups,upsilon

    def _make_rgrid(self, dr = 0.2,buf=2):
        self.dr = dr
        pos = np.array([a for a in self.magstruc.atoms])
        x_min = np.min(pos[:,0]) - buf
        x_max = np.max(pos[:,0]) + buf
        y_min = np.min(pos[:,1]) - buf
        y_max = np.max(pos[:,1]) + buf
        z_min = np.min(pos[:,2]) - buf
        z_max = np.max(pos[:,2]) + buf

        x = np.arange(x_min,x_max + dr, dr)
        y = np.arange(y_min,y_max + dr, dr)
        z = np.arange(z_min,z_max + dr, dr)
        N_x = len(x)
        N_y = len(y)
        N_z = len(z)
        X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
        
        rgrid = np.moveaxis([X,Y,Z],0,-1)
        
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
            self._make_rgrid()
        X = np.arange(self.rmin[0],stop = self.rmax[0]+self.dr,step = self.dr)
        Y = np.arange(self.rmin[1],stop = self.rmax[1]+self.dr,step = self.dr)
        Z = np.arange(self.rmin[2],stop = self.rmax[2]+self.dr,step = self.dr)

        return X,Y,Z


    def copy(self):
        '''
        Return a deep copy of the object
        '''
        raise NotImplementedError()
