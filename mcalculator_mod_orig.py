#!/usr/bin/env python
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve

'''
xyz: 2d array (3*n), position of spins
sxyz: 2d array (3*n), direction of spins
uclist: 1d array, m, a list of spin to calculate the mPDF,
    example: [10, 20, 30] -> calculate the mPDF of no.10, no.20, no.30 spin in the list
qgrid: 1d array, Q grid of IQ
rstep, rmax: float, define the r grid of PDF 
'''
def j0calc(q,params):
    '''
    Module to calculate the magnetic form factor j0 based on the tabulated analytical approximations.
    
    Inputs: array q giving the q grid, and list params, containing the coefficients for the analytical approximation to j0 as contained in tables by Brown.
    
    Returns: array with same shape as q giving the magnetic form factor j0.
    '''
    [A,a,B,b,C,c,D] = params
    return A*np.exp(-a*(q/4/np.pi)**2)+B*np.exp(-b*(q/4/np.pi)**2)+C*np.exp(-c*(q/4/np.pi)**2)+D

def calculateIQ(xyz, sxyz, uclist, qgrid, rstep, rmax, f):
    #qgrid = np.arange(qmin, qmax, qstep)
    S=np.linalg.norm(sxyz[0])
    r = np.arange(0, rmax, rstep)
    rbin =  np.concatenate([r-rstep/2, [r[-1]+rstep/2]])
    
    s1 = np.zeros(len(r))
    s2 = np.zeros(len(r))
    
    for i in range(len(uclist)):
        print 'Working on: '+str(i+1)+'/'+str(len(uclist))
        uu = uclist[i]
        
        ri = xyz0 = xyz[uu]
        rj = xyz
        si = sxyz0 = sxyz[uu]
        sj = sxyz
        
        dxyz = rj-ri
        d2xyz = np.sum((dxyz)**2, axis=1).reshape(dxyz.shape[0], 1)
        d1xyz = np.sqrt(d2xyz)
        d1xyzr = d1xyz.ravel()
        
        xh = dxyz / d1xyz
        xh[uu] = 0
        yh = si - xh * np.sum(si*xh, axis=1).reshape(dxyz.shape[0], 1)
        yh_dis = np.sum(yh**2, axis = 1)
        yh_ind = np.nonzero(np.abs(yh_dis)<1e-10)
        yh[yh_ind] = [0,0,0]
        
        aij = np.sum(si * yh, axis=1) * np.sum(sj * yh, axis=1) / yh_dis
        aij[yh_ind] = 0
        bij = 2 * np.sum(si * xh, axis=1) * np.sum(sj * xh, axis=1) - aij
        bij[uu] = 0
        
        s1 += np.histogram(d1xyzr, bins=rbin, weights=aij)[0]
        s2 += np.histogram(d1xyzr, bins=rbin, weights=bij)[0]
        
    rv = np.zeros_like(qgrid)
    #index non-zero s1 and s2
    inds1 = np.nonzero(s1)[0]
    inds2 = np.nonzero(s2)[0]
    for i in inds1:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s1[i] * np.sin(qxr)/(qxr)
    for i in inds2:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s2[i] * (np.sin(qxr)/qxr**3-np.cos(qxr)/qxr**2)
    rv=rv*(f**2)
    rv+=len(uclist)*2.*S*(S+1)*(f**2)/3.
    return [qgrid, rv]

def calculateIQPBC(xyz, sxyz, uclist, qgrid, rstep, f, latparams):
    S=np.linalg.norm(sxyz[0])
    x,y,z = np.transpose(xyz)
    nx,ny,nz=np.ceil((np.max(x)-np.min(x))/latparams[0]),np.ceil((np.max(y)-np.min(y))/latparams[1]),np.ceil((np.max(z)-np.min(z))/latparams[2]) ### assuming input list is cubic
    X,Y,Z=nx*latparams[0],ny*latparams[1],nz*latparams[2]
    boxsize=np.array([X,Y,Z])
    rmax=0.5*np.min(boxsize)
    print 'rmax='+str(rmax)
    r = np.arange(0, rmax, rstep)
    rbin =  np.concatenate([r-rstep/2, [r[-1]+rstep/2]])
    
    s1 = np.zeros(len(r))
    s2 = np.zeros(len(r))
    
    for i in range(len(uclist)):
        print 'Working on: '+str(i+1)+'/'+str(len(uclist))
        uu = uclist[i]
        
        ri = xyz0 = xyz[uu]
        rj = xyz
        si = sxyz0 = sxyz[uu]
        sj = sxyz
        
        dxyz = rj-ri
        dxyz = np.where(dxyz>0.5*boxsize,dxyz-boxsize,dxyz)
        dxyz = np.where(dxyz<-0.5*boxsize,dxyz+boxsize,dxyz)
        d2xyz = np.sum((dxyz)**2, axis=1).reshape(dxyz.shape[0], 1)
        d1xyz = np.sqrt(d2xyz)
        d1xyzr = d1xyz.ravel() ## need to maybe check that all distances are less than rmax?
        
        xh = dxyz / d1xyz
        xh[np.isnan(xh)] = 0
        yh = si - xh * np.sum(si*xh, axis=1).reshape(dxyz.shape[0], 1)
        yh_dis = np.sum(yh**2, axis = 1)
        yh_ind = np.nonzero(np.abs(yh_dis)<1e-10)
        yh[yh_ind] = np.array([0,0,0])
        
        aij = np.sum(si * yh, axis=1) * np.sum(sj * yh, axis=1) / yh_dis ## check and see why I am dividing by yh_dis
        aij[yh_ind] = 0
        bij = 2 * np.sum(si * xh, axis=1) * np.sum(sj * xh, axis=1) - aij
        bij[uu] = 0
        
        r_ind=np.nonzero(d1xyzr>rmax)
        aij[r_ind] = 0
        bij[r_ind] = 0
        
        s1 += np.histogram(d1xyzr, bins=rbin, weights=aij)[0]
        s2 += np.histogram(d1xyzr, bins=rbin, weights=bij)[0]
        
    rv = np.zeros_like(qgrid)
    #index non-zero s1 and s2
    inds1 = np.nonzero(s1)[0]
    inds2 = np.nonzero(s2)[0]
    for i in inds1:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s1[i] * np.sin(qxr)/(qxr)
    for i in inds2:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s2[i] * (np.sin(qxr)/qxr**3-np.cos(qxr)/qxr**2)
    rv=rv*(f**2)
    rv+=len(uclist)*2.*S*(S+1)*(f**2)/3.
    return [qgrid, rv]

    
def calculateIQPBCold(xyz, sxyz, uclist, qgrid, rstep, rmax, f, a):  ### a is lattice parameter of unit cell
    from cubicPBC import cubicPBC
    S=np.linalg.norm(sxyz[0])
    r = np.arange(0, rmax, rstep)
    rbin =  np.concatenate([r-rstep/2, [r[-1]+rstep/2]])
    
    s1 = np.zeros(len(r))
    s2 = np.zeros(len(r))
    
    xyz,sxyz=cubicPBC(xyz,sxyz,a)
    
    for i in range(len(uclist)):
        print 'Working on: '+str(i+1)+'/'+str(len(uclist))
        uu = uclist[i]
        
        ri = xyz0 = xyz[uu]
        rj = xyz
        si = sxyz0 = sxyz[uu]
        sj = sxyz
        
        dxyz = rj-ri
        d2xyz = np.sum((dxyz)**2, axis=1).reshape(dxyz.shape[0], 1)
        d1xyz = np.sqrt(d2xyz)
        d1xyzr = d1xyz.ravel()
        
        xh = dxyz / d1xyz
        xh[uu] = 0
        yh = si - xh * np.sum(si*xh, axis=1).reshape(dxyz.shape[0], 1)
        yh_dis = np.sum(yh**2, axis = 1)
        yh_ind = np.nonzero(np.abs(yh_dis)<1e-10)
        yh[yh_ind] = [0,0,0]
        
        aij = np.sum(si * yh, axis=1) * np.sum(sj * yh, axis=1) / yh_dis
        aij[yh_ind] = 0
        bij = 2 * np.sum(si * xh, axis=1) * np.sum(sj * xh, axis=1) - aij
        bij[uu] = 0
        
        s1 += np.histogram(d1xyzr, bins=rbin, weights=aij)[0]
        s2 += np.histogram(d1xyzr, bins=rbin, weights=bij)[0]
        
    rv = np.zeros_like(qgrid)
    #index non-zero s1 and s2
    inds1 = np.nonzero(s1)[0]
    inds2 = np.nonzero(s2)[0]
    for i in inds1:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s1[i] * np.sin(qxr)/(qxr)
    for i in inds2:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s2[i] * (np.sin(qxr)/qxr**3-np.cos(qxr)/qxr**2)
    rv=rv*(f**2)
    rv+=len(uclist)*2.*S*(S+1)*(f**2)/3.
    return [qgrid, rv]

    
def calculateMPDF(xyz, sxyz, uclist, rstep, rmax, psigma=0.05):
    #get s1, s2
    
    r = np.arange(0+rstep, rmax, rstep)
    rbin =  np.concatenate([r-rstep/2, [r[-1]+rstep/2]])
    
    s1 = np.zeros(len(r))
    s2 = np.zeros(len(r))
    
    for i in range(len(uclist)):
        uu = uclist[i]
        
        ri = xyz0 = xyz[uu]
        rj = xyz
        si = sxyz0 = sxyz[uu]
        sj = sxyz
        
        dxyz = rj-ri
        d2xyz = np.sum((dxyz)**2, axis=1).reshape(dxyz.shape[0], 1)
        d1xyz = np.sqrt(d2xyz)
        d1xyzr = d1xyz.ravel()
        
        xh = dxyz / d1xyz
        xh[uu] = 0
        yh = si - xh * np.sum(si*xh, axis=1).reshape(dxyz.shape[0], 1)
        yh_dis = np.sum(yh**2, axis = 1)
        yh_ind = np.nonzero(np.abs(yh_dis)<1e-10)
        yh[yh_ind] = [0,0,0]
        
        aij = np.sum(si * yh, axis=1) * np.sum(sj * yh, axis=1) / yh_dis
        aij[yh_ind] = 0
        bij = 2 * np.sum(si * xh, axis=1) * np.sum(sj * xh, axis=1) - aij
        bij[uu] = 0
        
        w2 = bij / d1xyzr**3
        w2[uu] = 0
        
        s1 += np.histogram(d1xyzr, bins=rbin, weights=aij)[0]
        s2 += np.histogram(d1xyzr, bins=rbin, weights=w2)[0]
    
    #shape function
    if psigma != None:
        x = np.arange(-3, 3, rstep)
        y = np.exp(-x**2 / psigma**2 / 2) * (1 / np.sqrt(2*np.pi) / psigma)
    
        s1[0] = 0
        s1 = fftconvolve(s1, y)
        s1 = s1[len(x)/2: -len(x)/2+1]
        
        s2 = fftconvolve(s2, y) * rstep
        s2 = s2[len(x)/2: -len(x)/2+1]
        
    ss2 = np.cumsum(s2)
    gr = s1 / r + r * (ss2[-1] - ss2)
    gr /= len(uclist)
    gr[0] = 0
    return [r, gr]
    

# def test():
    # strufile = 'cif/ni_sc.cif'
    # from mstructure import MStruAdapter
    # stru = MStruAdapter(stru = strufile, name='mstru', periodic = True, rmax = 30)
    # stru.extend2Rmax(50)
    # xyz = stru.xyz_cartn
    # sxyz = stru.sxyz
    # uclist = stru.uclist
    # r, gr = calculateMPDF(xyz, sxyz, uclist, 0.01, 30, psigma=0.1)
    
    # plt.figure(1)
    # plt.plot(r,gr)
    # plt.show()
    # return


def cv(x1,y1,x2,y2):
    '''
    Module to compute convolution of functions y1 and y2.
    
    Inputs: array y1, x1, y2, x2. Should have the same grid spacing to be safe.
    
    Returns: arrays ycv and xcv giving the convolution.
    '''
    dx=x1[1]-x1[0]
    ycv = dx*np.convolve(y1,y2,'full')
    xcv=np.linspace(x1[0]+x2[0],x1[-1]+x2[-1],len(ycv))

    return xcv,ycv
    
def costransform(q,fq,rmin=0.0,rmax=50.0,rstep=0.1): # does not require even q-grid
    '''
    Module to compute cosine Fourier transform of f(q). Uses direct integration rather than FFT and does not require an even q-grid.
    
    Inputs: array q (>=0 only), array f(q) to be transformed, optional arguments giving rmin, rmax, and rstep of output r-grid.
    
    Returns: arrays r and fr, where fr is the cosine Fourier transform of fq.
    '''
    lostep = int(np.ceil((rmin - 1e-8) / rstep))
    histep = int(np.floor((rmax + 1e-8) / rstep)) + 1
    r = np.arange(lostep,histep)*rstep
    qrmat=np.outer(r,q)
    integrand=fq*np.cos(qrmat)
    fr=np.sqrt(2.0/np.pi)*np.trapz(integrand,q)
    return r,fr

	
def getDiffData(fileNames,fmt='pdfgui'):
	for name in fileNames:
		if fmt=='pdfgui':
			allcols = np.loadtxt(name,unpack=True,comments='#',skiprows=14)
			r,grcalc,diff=allcols1[0],allcols1[1],allcols1[4]
			grexp = grcalc+diff
			np.savetxt(name[:-4]+'.diff',np.transpose((r,diff)))
		else:
			print 'This format is not currently supported.'
	
def test():
    print 'This is not a rigorous test.'
    return
    
if __name__=='__main__':
    test()