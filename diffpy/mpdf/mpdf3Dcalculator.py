import copy
import numpy as np
from scipy import signal as sig
from scipy.spatial import KDTree as KDT

def gauss(grid,s=0.5):
    """Generate a gaussian kernel of arbitrary size and density

    This function generates a 3D guassian kernel based on the input grid and the standard
    deviation chosen. This is a simple replacement for the form factors that will be
    implemented in the future.

    Args:
        grid (array): the 3D spatial coordinates for the gaussian kernel
        s (float): The standard deviation for the gaussian kernel

    """
    g = lambda point: 1/(s*np.sqrt((2*np.pi*s)**3))*np.exp(-1/2*(np.linalg.norm(point)/s)**2)
    return np.apply_along_axis(g,3,grid)

def vec_ac(a1,a2,delta,corr_mode="same"):
    """Correlate two 3D vector fields

    This function computes the autocorrelation of two 3D vector fields on regular 
    grids. The autocorrelation is computed for each vector component and then summed

    Args:
        a1 (array): The virst array to correlate
        a2 (array): The second array to correlate
        delta (float): the spacing between grid points
        corr_mode (string): The mode to use for the scipy correlation function
    """
    ac = sig.correlate(a1[:,:,:,0],a2[:,:,:,0],mode=corr_mode)*delta**3
    ac += sig.correlate(a1[:,:,:,1],a2[:,:,:,1],mode=corr_mode)*delta**3
    ac += sig.correlate(a1[:,:,:,2],a2[:,:,:,2],mode=corr_mode)*delta**3
    return ac

def vec_con(a1,a2,delta,conv_mode="same"):
    """Implement convolution for 3D vector fields

    This function implements a convolution function for 3D vector fields on regular 
    grids. The convolution is computed for each component of the vector fields and summed

    Args:
        a1 (array): The first array to convolve
        a2 (array): The second array to convolve
        delta (float): The grid spacing
        conv_mode (string): The mode to use for scipy convolution

    """
    con = sig.convolve(a1[:,:,:,0],a2[:,:,:,0],mode=conv_mode)*delta**3
    con += sig.convolve(a1[:,:,:,1],a2[:,:,:,1],mode=conv_mode)*delta**3
    con += sig.convolve(a1[:,:,:,2],a2[:,:,:,2],mode=conv_mode)*delta**3
    return con

def ups(grid):
    """A function to generat the Upsilon filter from Roth et.al. (2018)

    This function computes an kernel using the upsilon function defined in Roth et.al.
    (2018), https://doi.org/10.1107/S2052252518006590.

    Args:
        grid (array): the spatial grid over which the kernel is to be defined
    """
    g = lambda point: [0,0,0] if np.abs(np.linalg.norm(point)) <1e-6 else  point/np.linalg.norm(point)**4
    return np.apply_along_axis(g,3,grid)

class MPDF3Dcalculator:
    """Create an MPDF3Dcalculator object to help calculate 3D-mPDF functions

    This class is loosely modelled after the PDFcalculator cless in diffpy.
    At minimum, tie requires a magnetic structure with atoms and spins and will
    calculate the 3D-mPDF from that.

    Args:
        magstruc (MagStructure object): gives the information about the magnetic
            structure. Must have arrays of atoms and spins
        gaussPeakWidth (float): The width of the gaussian function that represents atoms
        label (string): Optional label from the MPDF3Dcalculator

    """

    def __init__(self, magstruc=None, gaussPeakWidth=0.5, label=""):
        if magstruc is None:
            self.magstruc = []
        else:
            self.magstruc = magstruc

        self.gaussPeakWidth = gaussPeakWidth
        self.label = label
        self.Nx = None
        self.Ny = None
        self.Nz = None
        self.dr = None

    def __repr__(self):
        if self.label == None:
            return "3DMPDFcalculator() object"
        else:
            return self.label +  ": 3DMPDFcalculator() object"

    def calc(self, verbose=False, dr=None):
        """Calculate the 3D magnetic PDF

        Args:
            verbose (boolean): indicates whether to output progress 
            dr (float): the grid spacing to use
        """

        if dr is not None:
            self.dr = dr
        self._makeRgrid()

        s_arr = np.zeros((self.Nx,self.Ny,self.Nz,3))
        if verbose :
            print("Setting up point spins")

        for i in range(len(self.magstruc.atoms)):
            idx = np.rint((self.magstruc.atoms[i] - self.rmin)/self.dr).astype(int) 
            s_arr[idx[0],idx[1],idx[2]] = self.magstruc.spins[i]

        if verbose:
            print("Setting up filter grid")

        filter_x = np.arange(-3,3+self.dr,self.dr)
        X,Y,Z = np.meshgrid(filter_x,filter_x,filter_x,indexing='ij')
        filter_grid = np.moveaxis([X,Y,Z],0,-1)

        if verbose:
            print("Making filters")

        gaussian = gauss(filter_grid,s=self.gaussPeakWidth)
        upsilon = ups(filter_grid)

        if verbose:
            print("Convolving spin array")

        s_arr[:,:,:,0] = sig.convolve(s_arr[:,:,:,0],gaussian,mode='same')*self.dr**3
        s_arr[:,:,:,1] = sig.convolve(s_arr[:,:,:,1],gaussian,mode='same')*self.dr**3
        s_arr[:,:,:,2] = sig.convolve(s_arr[:,:,:,2],gaussian,mode='same')*self.dr**3

        if verbose:
            print("Computing mpdf")

        mag_ups = vec_con(s_arr,upsilon,self.dr)
        self.mpdf = vec_ac(s_arr,s_arr,self.dr,"full")
        self.mpdf += -1/(np.pi**4)*sig.correlate(mag_ups,mag_ups,mode="full")*self.dr**3
        
        return 

    def _makeRgrid(self,dr = None,buf=0):
        """Set up bounds and intervals of the spatial grid to use

        Args:
            dr (float): the grid spacing to use
            buf (float): the space to include on either side of the 
                spin distribution
        """
        if dr is not None:
            self.dr = dr
        if self.dr is None:
            self.dr = 0.2
        pos = np.array([a for a in self.magstruc.atoms])
        x_min = np.min(pos[:,0]) - buf
        x_max = np.max(pos[:,0]) + buf
        y_min = np.min(pos[:,1]) - buf
        y_max = np.max(pos[:,1]) + buf
        z_min = np.min(pos[:,2]) - buf
        z_max = np.max(pos[:,2]) + buf

        x = np.arange(x_min,x_max + self.dr, self.dr)
        y = np.arange(y_min,y_max + self.dr, self.dr)
        z = np.arange(z_min,z_max + self.dr, self.dr)
        N_x = len(x)
        N_y = len(y)
        N_z = len(z)
        x_max = np.max(x)
        y_max = np.max(y)
        z_max = np.max(z)
        X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
        
        rgrid = np.moveaxis([X,Y,Z],0,-1)
        
        self.Nx = N_x
        self.Ny = N_y
        self.Nz = N_z
        self.rmin = np.array([x_min,y_min,z_min])
        self.rmax = np.array([x_max,y_max,z_max])
        return rgrid

    def plot(self):
        """Plot the 3D-mPDF

        ToDo: implement plotting, use Jacobs visualilze
        """
        raise NotImplementedError()

    def runChecks(self):
        """Runs bounds and compatibility checks for internal variables. 
            This should be called during __init__
         
        ToDo: implement for troubleshooting
        """        
        raise NotImplementedError()

    def rgrid(self):
        """Returns the spatial grid the 3D-mPDF is output on

        Generates the spatial grid for the 3D-mPDF when needed by
        the user
        """

        if self.dr is None:
            self._makeRgrid()
        X = np.arange(-(self.Nx-1)*self.dr,stop = (self.Nx-1)*self.dr+self.dr/2,step = self.dr)
        X[self.Nx-1] = 0
        
        Y = np.arange(-(self.Ny-1)*self.dr,stop = (self.Ny-1)*self.dr+self.dr/2,step = self.dr)
        Y[self.Ny-1] = 0
        
        Z = np.arange(-(self.Nz-1)*self.dr,stop = (self.Nz-1)*self.dr+self.dr/2,step = self.dr)
        Z[self.Nz-1] = 0
        

        return X,Y,Z


    def copy(self):
        '''
        Return a deep copy of the object
        '''
        return copy.deepcopy(self)
