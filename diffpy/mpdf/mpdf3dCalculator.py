import numpy as np
from scipy import signal as sig
from scipy.spatial import KDTree as KDT

def gauss(grid,s=0.5):
    '''
    A function to generate a gaussian kernel of arbitrary size and density
    '''
    raise NotImplementedError()

def vec_ac(a1,a2,delta):
    '''
    A function to implement the autocorrelation for two vector fields
    '''
    raise NotImplementedError()

def vec_con(a1,a2,delta):
    '''
    A function to implement the convolution operator for two discrete vector fields
    '''
    raise NotImplementedError()

def ups(r):
    '''
    A function to generat the Upsilon filter from Roth et.al.
    '''
    raise NotImplementedError()

class 3dMPDFcalculator:

    def __init__():
        '''
        Need to decide on cononical input format
        '''
        raise NotImplementedError()

    def __repr__(self):
        if self.label == None:
            return "3dMPDFcalculator() object"
        else:
            return self.label + ": 3dMPDFcalculator() object"

    def calc(self):
        '''
        Calculate and store the 3DMPDF 
        '''
        raise NotImplementedError()

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
        raise NotImplementedError()


    def copy(self):
        '''
        Return a deep copy of the object
        '''
        raise NotImplementedError()
