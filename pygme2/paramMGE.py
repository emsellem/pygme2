#!/usr/bin/python
"""
This module reads and writes the parameters of a Multi Gaussian Expansion model (Monnet et al.
1992, Emsellem et al. 1994). It can read and write MGE input ascii files and
computes a number of basic parameters for the corresponding models.  

uptdated regularly and may still contains some obvious bugs. A stable version will
be available hopefully before the end of 2012.
For questions, please contact Eric Emsellem at eric.emsellem@eso.org
"""

"""
Importing the most import modules
This MGE module requires NUMPY and SCIPY
"""

import os

try:
    import numpy as np
except ImportError:
    raise Exception("numpy is required for pygme")

try:
    from scipy import special
except ImportError:
    raise Exception("scipy is required for pygme")

from numpy import asarray
from numpy import cos, sin, copy, sqrt, exp


__version__ = '1.1.6 (22 Dec 2014)'

## Version 1.1.6 : EE - Fixed found2D
## Version 1.1.5 : EE - Fixed mcut input parameter
## Version 1.1.4 : EE - Fixed a typo on indices
## Version 1.1.3 : EE - Added BetaEps, M/L etc also in the 2D Gauss just in case
## Version 1.1.2 : EE - Changed imin,imax into ilist
## Version 1.1.1 : EE - Removed the condition for comp_Nparticles when reading an mge
## Version 1.1.0 : EE - Some serious cleanup in the naming of the variables
## Version 1.0.2 : EE - few minor changes including adding saveMGE
## Version 1.0.1 : EE - replaces ones to zeros in initialisation of GaussGroupNumber

############################################################################
# Class to define dynamical MGE parameters useful for calculation purposes #
############################################################################
class dynParamMGE():
    """ 
    Class to add some parameters which are useful for dynamical routines
    """
    def __init__(self, MGEmodel):
        """ 
        Initialisation of the additional dynamical parameters
        """
        if (MGEmodel._model3d is not None):
            self.sig3d2_soft = MGEmodel.sig3d**2  + MGEmodel.Softarc**2  # Sigma softened in arcsec
            self.dsig3d2_soft = 2. * self.sig3d2_soft
            # Deriving some more numbers
            self.Bij = np.zeros((MGEmodel.n_gaussians, MGEmodel.n_gaussians), np.float32)
            self.Bij_soft = np.zeros((MGEmodel.n_gaussians, MGEmodel.n_gaussians), np.float32)
            self.e2q2dsig3d2 = np.zeros(MGEmodel.n_gaussians, np.float32)
            self.e2q2sig3d2 = np.zeros(MGEmodel.n_gaussians, np.float32)
            self.sqpi2s = sqrt(np.pi / 2.) / MGEmodel.qsig3d
            self.qq2s2 = 4. * MGEmodel.qzx2 * MGEmodel.sig3d2
            self.q2sig3d2 = MGEmodel.qzx2 * MGEmodel.sig3d2
            for i in range(MGEmodel.n_gaussians) :
                if self.q2sig3d2[i] != 0. :
                    self.e2q2dsig3d2[i] = MGEmodel.e2[i] / (2. * self.q2sig3d2[i])
                    self.e2q2sig3d2[i] = MGEmodel.e2[i] / self.q2sig3d2[i]
                else :
                    print "WARNING: %d component has q2*Sig2=0" %(i+1)
                for j in range(MGEmodel.n_gaussians) :
                    self.Bij[i,j] = MGEmodel.e2[j] - self.q2sig3d2[i] / MGEmodel.sig3d2[j]
                    self.Bij_soft[i,j] = MGEmodel.e2[j] - self.q2sig3d2[i] / self.sig3d2_soft[j]

            self.kRZ2 = MGEmodel.kRZ**2
            self.mkRZ2q2 = 1. - self.kRZ2 * MGEmodel.qzx2
            self.mkRZ2 = 1. - self.kRZ2
            self.Dij = np.zeros((MGEmodel.n_gaussians,MGEmodel.n_gaussians), np.float32)
            self.Dij_soft = np.zeros((MGEmodel.n_gaussians,MGEmodel.n_gaussians), np.float32)
            for i in range(MGEmodel.n_gaussians) :
                for j in range(MGEmodel.n_gaussians) :
                    self.Dij[i,j] = self.mkRZ2[i] * self.Bij[i,j] + MGEmodel.e2[j] * self.kRZ2[i]
                    self.Dij_soft[i,j] = self.mkRZ2[i] * self.Bij_soft[i,j] + MGEmodel.e2[j] * self.kRZ2[i]

## ===========================================================================================

############################################################################
# Class to define photometric MGE parameters useful for calculation purposes #
############################################################################
class photParamMGE():
    """ 
    Class to add some parameters which are useful for photometric routines
    """
    def __init__(self, MGEmodel):
        """ 
        Initialisation of the additional photometric parameters
            These are hidden in this class
        """
        if (MGEmodel._model3d is not None):
            self.dsig3d = sqrt(2.) * MGEmodel.sig3d
            self.dsig3d2 = 2. * MGEmodel.sig3d2
            self.qParc = MGEmodel.qzx * MGEmodel.pa
            self.dqsig3d = sqrt(2.) * MGEmodel.qsig3d

## ===========================================================================================

class paramMGE() :
    def __init__(self, infilename=None, saveMGE=None, indir=None, **kwargs) :
        """
        Initialisation of the MGE model - reading the input file

        infilename : input MGE ascii file defining the MGE model
        indir: directory where to find the mge file
        saveMGE: directory in which some MGE model will be saved automatically during the
                 realisation of the Nbody sample
                 If saveMGE is None (default), it will be defined as ~/MGE
                 This will be created by default (if not existing)
                 
        Additional Input (not required):
            nTotalPart: total number of particles
            nPartStar : number of Stellar particles
            nPartHalo: number of Dark Matter particles
            nPartGas : number of Gas particles

            FirstRealisedPart : number for the first realised Particle 
                                This is useful if we wish to realise the model in chunks
            nMaxPart : Max number of particles to be realised for this run

            mcut : cut in pc, Default is 50 000 (50 kpc)
                   Used for the Ellipsoid truncation

            Rcut : cut in pc, Default is 50 000 (50 kpc)
            Zcut : cut in pc, Default is 50 000 (50 kpc)
                   Used for the Cylindre truncation
 
            FacBetaEps : Coefficient for : Beta = Coef * Epsilon
                         Default if Coef = 0.6
                         Can also be a vector (one for each Gaussian)

            MaxFacBetaEps: maximum value allowed for FacBetaEps. Default is 0.8.

        """

        ## Now checking if saveMGE has been defined and act accordingly
        if saveMGE is None :
            ## This is the default dir (~/MGE) if none is given
            saveMGE = os.path.expanduser("~/MGE")
            if not os.path.isdir(saveMGE) :
                ## Creating the default saveMGE directory
                os.system("mkdir ~/MGE")

        ## Test now if this exists
        if not os.path.isdir(saveMGE) :
            print "ERROR: directory for Archival does not exist = %s"%(saveMGE)
            return
        ## Finally save the value of saveMGE in the structure
        self.saveMGE = saveMGE

        ## Setting up some fixed variable #####################################
        ## G is in (km/s)2. Msun-1 . pc .
        ## OLD VALUE WAS:  self.Gorig = 0.0043225821
        self.Gorig = np.float32(0.0043225524) # value from Remco van den Bosch

        self.nPart = np.int(kwargs.get("nTotalPart", 0))   # TOTAL Number of n bodies
        self.nPartStar = np.int(kwargs.get("nPartStar", 0))   # TOTAL Number of n bodies
        self.nPartHalo = np.int(kwargs.get("nPartHalo", 0))   # TOTAL Number of n bodies
        self.nPartGas = np.int(kwargs.get("nPartGas", 0))   # TOTAL Number of n bodies
        self.Add_BHParticle = True   # Add a BH if Mbh > 0 when realising particles

        self.FirstRealisedPart = np.int(kwargs.get("FirstRealisedPart", 0))   # First Realised Particle
        self.nMaxPart = np.int(kwargs.get("nMaxPart", 0))   # Max number of particles to be realised

        # Viewing angles in degrees
        # Second value is inclination for axisymmetric systems
        # 90 degrees inclination means edge-on view
        self.euler_angles = kwargs.get("euler_angles", np.array([0., 90., 0.]))
        
        # Truncation Method = Default is Ellipsoid, (can also be Cylindre)
        self.truncation_method = kwargs.get("truncation_method", "Ellipsoid")   
        self.mcut = kwargs.get("Mcut", 50000.)   # Default truncation in pc - Default is 50kpc
        self.Rcut = kwargs.get("Rcut", 50000.)   # Default truncation in pc - Default is 50kpc
        self.Zcut = kwargs.get("Zcut", 50000.)   # Default truncation in pc - Default is 50kpc

        self.Mbh = 0.           # Black hole mass
        self.axi = 1

        self.Nquad = 100              # Number of Points for the Quadrature, default is 100

        self.FacBetaEps = kwargs.get("FacBetaEps", 0.6)   # Coefficient for the BETAEPS option: Beta = Coef * Epsilon
        self.MaxFacBetaEps = kwargs.get("MaxFacBetaEps", 0.8)   # Max value the BETAEPS Factor
        self.DummyFacBetaEps = 0.6

        ## Test if infilename is None. If this is the case reset MGE with 0 Gaussians
        self.n_gaussians = self.n_group = self.n_dyncomp = 0
