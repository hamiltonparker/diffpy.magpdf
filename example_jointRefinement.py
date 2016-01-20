#!/usr/bin/env python
# -*- coding: utf-8 -*-


# We'll need numpy and pylab for plotting our results
import numpy as np
#import pylab
import matplotlib.pyplot as plt

# A least squares fitting algorithm from scipy
from scipy.optimize.minpack import leastsq

# DiffPy-CMI modules for building a fitting recipe
from diffpy.Structure import loadStructure
from diffpy.srfit.pdf import PDFContribution
from diffpy.srfit.fitbase import FitRecipe, FitResults

# Load the mPDF calculator modules
from mcalculator import *


# Files containing our experimental data and structure file
dataFile = "mPDF_exampleFiles/npdf_07334.gr"
structureFile = "mPDF_exampleFiles/MnO_R-3m.cif"
spaceGroup = "R-3m"

# The first thing to construct is a contribution. Since this is a simple
# example, the contribution will simply contain our PDF data and an associated
# structure file. We'll give it the name "nickel"
MnOPDF = PDFContribution("MnO")

# Load the data and set the r-range over which we'll fit
MnOPDF.loadData(dataFile)
MnOPDF.setCalculationRange(xmin=0.01, xmax=20, dx=0.01)

# Add the structure from our cif file to the contribution
MnOStructure = loadStructure(structureFile)
MnOPDF.addStructure("MnO", MnOStructure)

# Set up the mPDF calculator
mc=mPDFcalculator(MnOStructure,magIdxs=[0,1,2],rmin=0.01,rmax=20.0,rstep=0.01,gaussPeakWidth=0.2)
mc.svec=2.5*np.array([1.0,-1.0,0])/np.sqrt(2)
mc.kvec=np.array([0,0,1.5])
mc.spinOrigin=np.array([0,0,0])
mc.ffqgrid=np.arange(0,10,0.01)
mc.ff=jCalc(mc.ffqgrid,getFFparams('Mn2'))
mc.calcList=np.arange(1)

# The FitRecipe does the work of calculating the PDF with the fit variable
# that we give it.
MnOFit = FitRecipe()

# give the PDFContribution to the FitRecipe
MnOFit.addContribution(MnOPDF)

# Configure the fit variables and give them to the recipe.  We can use the
# srfit function constrainAsSpaceGroup to constrain the lattice and ADP
# parameters according to the R-3m space group.
from diffpy.srfit.structure import constrainAsSpaceGroup
spaceGroupParams = constrainAsSpaceGroup(MnOPDF.MnO.phase, spaceGroup)
print "Space group parameters are:",
print ', '.join([p.name for p in spaceGroupParams])
print

# We can now cycle through the parameters and activate them in the recipe as
# variables
for par in spaceGroupParams.latpars:
    MnOFit.addVar(par)
# Set initial value for the ADP parameters, because CIF had no ADP data.
for par in spaceGroupParams.adppars:
    MnOFit.addVar(par, value=0.003,fixed=True)

# As usual, we add variables for the overall scale of the PDF and a delta2
# parameter for correlated motion of neighboring atoms.
MnOFit.addVar(MnOPDF.scale, 1)
MnOFit.addVar(MnOPDF.MnO.delta2, 1.5)

# We fix Qdamp based on prior information about our beamline.
MnOFit.addVar(MnOPDF.qdamp, 0.03, fixed=True)
ginit=MnOFit.MnO.evaluate()
# Turn off printout of iteration number.
MnOFit.clearFitHooks()

# We can now execute the fit using scipy's least square optimizer.

# Initial structural fit
print "Refine PDF using scipy's least-squares optimizer:"
print "  variables:", MnOFit.names
print "  initial values:", MnOFit.values
leastsq(MnOFit.residual, MnOFit.values)
print "  final values:", MnOFit.values
print
# Obtain and display the fit results.
MnOResults = FitResults(MnOFit)
print "FIT RESULTS\n"
print MnOResults

# Initial mPDF fit
mc.struc=MnOStructure
mc.makeAtoms()
mc.makeSpins()

gobs=MnOFit.MnO.profile.y

def magresidual(p,yexp,mcalc):
    mcalc.paraScale,mcalc.ordScale=p
    return yexp-mcalc.calc(both=True)[2]

print 'Perform initial refinement of mPDF:'
p0=[5.0,3.0]
pOpt=leastsq(magresidual,p0,args=(gobs-MnOFit.MnO.evaluate(),mc))
print pOpt

def jointResidual(p): #not really working; doesn't crash, but doesn't really refine either
    lata,latc,scalePDF,delta2,paraScale,ordScale=p
    mc.struc=MnOStructure    
    mc.paraScale=paraScale
    mc.ordScale=ordScale
    mc.makeAtoms()
    mc.makeSpins()
    return gobs-MnOPDF.evaluate()-mc.calc(both=True)[2]

print "Joint refinement of structural and magnetic PDF:"
#MnOStructure.lattice.c=7.63
pJoint=leastsq(jointResidual, np.concatenate((MnOFit.values,pOpt[0])))
print "  final values:", pJoint[0]
print


# Plot the observed and refined PDF.

# Get the experimental data from the recipe
r = MnOFit.MnO.profile.x
gobs = MnOFit.MnO.profile.y

# Get the calculated PDF and compute the difference between the calculated and
# measured PDF
gcalc = MnOFit.MnO.evaluate()
baseline = 1.1 * gobs.min()
gdiff = gobs - gcalc
baseline2 = 1.1 * (gdiff+baseline).min()
magfit=mc.calc(both=True)[2]

# Plot!
ax=plt.figure().add_subplot(111)
ax.plot(r, gobs, 'bo', label="G(r) data",markerfacecolor='none', markeredgecolor='b')
ax.plot(r, gcalc, 'r-', lw=1.5, label="G(r) fit")
ax.plot(r, gdiff + baseline,mfc='none',mec='b',marker='o')
ax.plot(r,magfit+baseline,'r-',lw=1.5)
ax.plot(r,gdiff-magfit+baseline2,'g-')
ax.plot(r, np.zeros_like(r) + baseline2, 'k:')
ax.set_xlabel(r"r ($\AA$)")
ax.set_ylabel(r"G ($\AA^{-2}$)")
plt.legend()

plt.show()
