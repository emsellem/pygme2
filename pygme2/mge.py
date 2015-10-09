# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""MGE modelling module

This module includes the main MGE class inheriting from the BaseModel
For questions, please contact Eric Emsellem at eric.emsellem@eso.org

This module requires astropy - Raises an Exception if not available
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2015, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# Importing Modules
import numpy as np

#from rwcfor import np.float32
#from mge_miscfunctions import print_msg
import os

from BaseMGE import BaseMGEModel, BaseMultiGaussian2D
from dynMGE import dynMGE
from paramMGE import dynParamMGE, photParamMGE, paramMGE

try:
    import astropy as apy
    from astropy.modeling import models as astropy_models
    from astropy.modeling import Parameter, Model, Fittable1DModel
    from astropy import constants as constants, units as units
except ImportError:
    raise Exception("astropy is required for this module")

__version__ = '0.0.1 (14 August 2014)'

class MGEModel(BaseMGEModel, dynMGE, paramMGE) :
    """ MGE model 

    This class defines the basic MGE model, which should include both
    a reference 2D Base model made of n_gaussians Gaussians, and the
    associated 3D Gaussians, using the viewing Euler Angles
    """

    def __init__(self, infilename=None, indir=None, **kwargs) :
        """Initialise the MGE model
        """

        # General verbose parameter for the MGE model
        self.verbose = kwargs.get("verbose", False)

        # Initial value for the Gravitational Constant
        # G is in (km/s)2. Msun-1 . pc .
        self.GGRAV = constants.G.to(units.km**2 * units.pc 
                         / units.s**2 / units.M_sun).value
        # Distance of the model in parsec
        self.distance = kwargs.get("distance", 1.0e6)

        # Input Parameters
        self.n_gaussians = np.int(kwargs.get("n_Gaussians", 1))
        
        # Now setting the 2D / 3D Gaussians
        BaseMGEModel.__init__(self, **kwargs)
        
        # add the additional parameters
        self._add_Parameters()
        
        # setting up the dynamical model
        dynMGE.__init__(self)
        
        if infilename is not None :
            self.read_mge(infilename=infilename, indir=indir)
        else :
            print "you need to specify infilename"
            return

    # =================================================
    # Distance is a property which defines the scale
    # including the parsec per arsec scale
    # =================================================
    @property
    def distance(self) :
        return self._distance

    @distance.setter
    def distance(self, value) :
        if value is None :
            if self.verbose : print("WARNING: setting default" 
                              "Distance to 10.0 Mpc (10^6 pc)")
            # Setting the default in case the Distance is None
            value = 1.0e6 
        elif value <= 0. :
            if self.verbose:
                print("WARNING: you provided a negative Distance value")
                print("WARNING: it will be set to the default (10 Mpc)")
            # Setting the default in case the Distance is negative
            value = 1.0e6 

        # Deriving the scale conversion factor between pc and arcsec
        self._pc_per_arcsec = np.float32(np.pi * value / 648000.0)
        # Default truncation - in arcseconds at 10 Mpc
        self._mcutarc = self.mcut / self._pc_per_arcsec   

        # Gravitation constant is in (km/s)2. Msun-1 . pc
        # We need to include arcseconds in there to deal 
        # with MGE models (which depends on distance)
        # We multiply it by pc . arcsec-1
        # so the unit becomes:  (km/s)2. Msun-1 . pc2 . arcsec-1
        self._GGRAV_arc = self.GGRAV * self._pc_per_arcsec
        # We now calculate 4 * PI * G in units of Garc
        self._PIG = 4. * np.pi * self._GGRAV_arc

        self._distance = value
    # --------------------------------------------------
    
    def getVescpae(self, R, Z, ilist=None):
        self.Vescape(R, Z, ilist)
    
    def _reset(self) :
        self.n_gaussians = 0
        self.euler_angles[0] = self.euler_angles[1] = self.euler_angles[2] = 0
        self._model2d = None
        self._model3d = None

    ##################################################################
    ### Write an ascii MGE file using an existing MGE class object ###
    ##################################################################
    def write_mge(self, outdir=None, outfilename=None, overwrite=False) :
        if (outfilename is None) :                       # testing if the name was set
            print 'You should specify an output file name'
            return

        if outdir is not None :
            outfilename = outdir + outfilename

        ## Testing if the file exists
        if os.path.isfile(outfilename) :
            if not overwrite : # testing if the existing file should be overwritten
                print 'WRITING ERROR: File %s already exists, use overwrite=True if you wish' %outfilename
                return

        mgeout = open(outfilename, "w+")
        ## Starting to write the output file
        linecomment = "#######################################################\n"

        def set_txtcomment(text, name, value, valform="%f") :
            textout = "## %s \n"%(text)
            return textout + name + " " + valform%(value)+"\n"

        mgeout.write(linecomment + "## %s MGE model \n"%(outfilename) + linecomment)

        ## Basic Parameters
        mgeout.write(set_txtcomment("Distance [Mpc]", "DIST", self.distance, "%5.2f"))
        mgeout.write(set_txtcomment("Euler Angles [Degrees]", "EULER", tuple(self.euler_angles), "%8.5f %8.5f %8.5f"))

        ## Number of Gaussians
        mgeout.write(set_txtcomment("Number of Gaussians", "NGAUSS", self.n_gaussians, "%d"))

        ###################
        ## 2D Gaussians
        ###################
        ## STARS First
        mgeout.write("## No                  Imax   Sigma      Q      PA\n")
        for i in range(self.n_gaussians) :
            mgeout.write("GAUSS2D%02d   "%(i) + "%8.5e %8.5f %8.5f %8.5f \n"%(self.imax2d[i], self.sig2d[i], self.q2d[i], self.pa[i]))
     
        ###################
        ## 3D Gaussians
        ###################
        mgeout.write("## ID                  Imax    Sigma       qzx       QyZ   \n")
        for i in range(self.n_gaussians) :
            mgeout.write("GAUSS3D%02d   "%(i) + "%8.5e %8.5f %8.5f %8.5f \n"%(self.imax3d[i], self.sig3d[i], self.qzx[i], self.qzy[i]))

        mgeout.close()
#===================================================================================================================================

    ##################################################################
    ### Reading an ascii MGE file and filling the MGE class object ###
    ##################################################################
    def read_mge(self, infilename=None, indir=None) :

        if (infilename is not None) :                       # testing if the name was set
            if indir is not None :
                infilename = indir + infilename

            if not os.path.isfile(infilename) :          # testing the existence of the file
                print 'OPENING ERROR: File %s not found' %infilename
                return
            
            ## resets n_gaussians to 0 and model2d and model3d to None
            self._reset()

            ################################
            # Opening the ascii input file #
            ################################
            self.pwd = os.getcwd()
            self.fullMGEname = os.path.abspath(infilename)
            self.MGEname = os.path.basename(self.fullMGEname)
            self.pathMGEname = os.path.dirname(self.fullMGEname)

            mge_file = open(self.fullMGEname)

            lines = mge_file.readlines()
            nlines = len(lines)

            ########################################
            ## First get the Number of gaussians  ##
            ## And the global set of parameters   ##
            ########################################
      
            for i in xrange(nlines) :
                if lines[i][0] == "#" or lines[i] == "\n" :
                    continue
                sl = lines[i].split()
                keyword = sl[0]
                if (keyword[:6] == "NGAUSS") :
                    nGauss = np.int(sl[1])
                    self.n_gaussians(nGauss)
                    keynGauss = 1
                elif (keyword[:4] == "DIST") :
                    Dist = np.float32(sl[1])
                    self.distance(Dist)
             
            tmpimax2d = [0.0]
            tmpsig2d = [0.0]
            tmpq2d = [0.0]
            tmppa = [0.0]
            
            tmpimax3d = [0.0] 
            tmpsig3d = [0.0]   
            tmpqzx = [0.0]   
            tmpqzy = [0.0]                           
            ##================================================================================##
            ## Then really decoding the lines and getting all the details from the ascii file ##
            ##================================================================================##
            for i in xrange(nlines) :
                if (lines[i][0] == "#")  or (lines[i] == "\n") :
                    continue
                sl = lines[i].split()
                keyword = sl[0]
                if (keyword[:6] == "NGAUSS") or (keyword[:4] == "DIST"):
                    continue
                ## projected gaussians
                elif (keyword[:11] == "GAUSS2D") :
                    tmpimax2d.append(sl[1])
                    tmpsig2d.append(sl[2])
                    tmpq2d.append(sl[3])
                    tmppa.append(sl[4])
                ## spacial gaussians
                elif (keyword[:11] == "GAUSS3D") :  
                    tmpimax3d.append(sl[1])
                    tmpsig3d.append(sl[2])
                    tmpqzx.append(sl[3])
                    tmpqzy.append(sl[4])              
                ## Center and other parameters
                elif (keyword[:5] == "EULER") :
                    self.euler_angles = np.zeros((3,), dtype=np.float32)
                    self.euler_angles[0] = np.float32(sl[1])
                    self.euler_angles[1] = np.float32(sl[2])
                    self.euler_angles[2] = np.float32(sl[3])
                else :
                    print 'Could not decode the following keyword: %s' %keyword
                    mge_file.close
                    break
            ################################
            # CLOSING the ascii input file #
            ################################
            mge_file.close

            tmpimax2d.pop(0)
            tmpsig2d.pop(0)
            tmpq2d.pop(0)
            tmppa.pop(0) 
        
            tmpimax3d.pop(0)
            tmpsig3d.pop(0)
            tmpqzx.pop(0)
            tmpqzy.pop(0)
            
            tmpmodel2d = BaseMultiGaussian2D(tmpimax2d,tmpsig2d,tmpq2d,pa=tmppa)
            self.model2d = tmpmodel2d
        
        # no name was specified #
        else :
            print 'You should specify an output file name'

    #====================== END OF READING / INIT THE MGE INPUT FILE =======================#
    
    #####################################
    ## Adding more Gaussian parameters ##
    #####################################
    def _add_Parameters(self) :
        """
        Add many more parameters using the basic I, Sig, q, PA parameters of the model
        These parameters are important for many (photometry/dynamics-related) routines
        """

        ## Only if axisymmetric
        if self.axi :

            ##################################################################
            ## Compute some useful parameters for the projected Gaussians
            ##################################################################
            if  (self._model2d is not None) :
                # some useful numbers from the projected gaussians if they exist
                #!TODO sigma units??? 
                self.sig2dpc =  self.sig2d * self._pc_per_arcsec        # Sigma in pc
                self.q2d2 = self.q2d * self.q2d
                self.sig2d2 = self.sig2d * self.sig2d  # Projected Sigma in arcsecond
                self.dsig2d2 = 2. * self.sig2d2
                self.Pp = self.imax2d * self.ML      # Mass maximum in Mass/pc-2
                self.MGEFluxp = self.imax2d*(self.sig2dpc**2) * self.q2d2 * np.pi

            ##################################################################
            ## Compute some useful parameters for the Spatial Gaussians
            ##################################################################
            if (self._model3d is not None):
                # some more useful numbers
                #!TODO sigma and imax units??? 
                self.imax3dpc = self.imax3d / self._pc_per_arcsec  # I in Lum.pc-3
                self.sig3dpc =  self.sig3d * self._pc_per_arcsec # Sigma in pc
                #!TODO imax3dpc???
                self.Parc = self.imax3d * self.ML  # Mass maximum in Mass/pc-2/arcsec-1
                self.qzx2 = self.qzx ** 2
                self.e2 = 1. - self.qzx2
                self.sig3d2 = self.sig3d**2           # Sigma in arcsecond !
                self.qsig3d = self.qzx * self.sig3d

                ## Add photometric parameters
                self._pParam = photParamMGE(self)
                ## Add dynamics parameters
                self._dParam = dynParamMGE(self)

                ## Fluxes and Masses
                self.MGEFlux = self.imax3dpc * self.qzx * (np.sqrt(2.*np.pi) * self.sig3dpc)**3
                self.MGEMass = self.MGEFlux * self.ML

#                 ## Total Mass and Flux for Stars and Gas and Halo (not truncated)
#                 self.MGEStarMass = np.sum(self.MGEMass[:self.nStarGauss],axis=0)
#                 self.MGEStarFlux = np.sum(self.MGEFlux[:self.nStarGauss],axis=0)
#                 self.MGEGasMass = np.sum(self.MGEMass[self.nStarGauss:self.nStarGauss+self.nGasGauss],axis=0)
#                 self.MGEGasFlux = np.sum(self.MGEFlux[self.nStarGauss:self.nStarGauss+self.nGasGauss],axis=0)
#                 self.MGEHaloMass = np.sum(self.MGEMass[self.nStarGauss+self.nGasGauss:self.nStarGauss+self.nGasGauss+self.nHaloGauss],axis=0)
#                 self.MGEHaloFlux = np.sum(self.MGEFlux[self.nStarGauss+self.nGasGauss:self.nStarGauss+self.nGasGauss+self.nHaloGauss],axis=0)
#                 ## Total Mass and Flux for all
#                 self.TMGEFlux = np.sum(self.MGEFlux,axis=0)
#                 self.TMGEMass = np.sum(self.MGEMass,axis=0)
# 
#                 self.facMbh = self.Mbh / (4. * np.pi * self._pc_per_arcsec * self._pc_per_arcsec)  # in M*pc-2*arcsec2
# 
#                 ## TRUNCATED Mass and Flux for each Gaussian
#                 self.truncMass = np.zeros(self.nGauss, np.float32)
#                 self.truncFlux = np.zeros(self.nGauss, np.float32)
#                 if self.TruncationMethod == "Cylindre" :
#                     for i in range(self.nGauss) :
#                         self.truncFlux[i] = self.rhointL_1G(self.Rcutarc, self.Zcutarc, i)
#                         self.truncMass[i] = self.rhointM_1G(self.Rcutarc, self.Zcutarc, i)
#                 elif  self.TruncationMethod == "Ellipsoid" :
#                     for i in range(self.nGauss) :
#                         self.truncFlux[i] = self.rhoSphereintL_1G(self.mcutarc, i)
#                         self.truncMass[i] = self.rhoSphereintM_1G(self.mcutarc, i)
#                 ## Total TRUNCATED Flux and Mass
#                 self.TtruncFlux = np.sum(self.truncFlux,axis=0)
#                 self.TtruncMass = np.sum(self.truncMass,axis=0)
# 
#                 # Listing the Gaussians in the Groups
#                 self._listGroups()
#                 self._listDynComps()
# 
#                 ## Total Mass and Flux for Groups TRUNCATED!
#                 self.truncGroupMass = np.zeros(self.nGroup, np.float32)
#                 self.truncGroupFlux = np.zeros(self.nGroup, np.float32)
#                 for i in range(self.nGroup) :
#                     self.truncGroupMass[i] = np.sum(self.truncMass[self.listGaussGroup[i]], axis=0)
#                     self.truncGroupFlux[i] = np.sum(self.truncFlux[self.listGaussGroup[i]], axis=0)
#                 ## Total TRUNCATED Flux and Mass for STARS, GAS, HALO
#                 ## STARS
#                 self.truncStarFlux = np.sum(self.truncFlux[0: self.nStarGauss])
#                 self.truncStarMass = np.sum(self.truncMass[0: self.nStarGauss])
#                 ## GAS
#                 self.truncGasFlux = np.sum(self.truncFlux[self.nStarGauss:self.nStarGauss + self.nGasGauss])
#                 self.truncGasMass = np.sum(self.truncMass[self.nStarGauss:self.nStarGauss + self.nGasGauss])
#                 ## HALO
#                 self.truncHaloFlux = np.sum(self.truncFlux[self.nStarGauss + self.nGasGauss:self.nStarGauss + self.nGasGauss + self.nHaloGauss])
#                 self.truncHaloMass = np.sum(self.truncMass[self.nStarGauss + self.nGasGauss:self.nStarGauss + self.nGasGauss + self.nHaloGauss])

        else :
            print "Triaxial model, cannot compute additional photometric parameters"

    ## ===========================================================================================================

def create_mge(outfilename=None, overwrite=False, outdir=None, **kwargs) :
    """Create an MGE ascii file corresponding to the input parameters
    """
    
    tmpMGE = MGEModel(**kwargs)
    
    tmpMGE.write_mge(outfilename,overwrite,outdir)
        
    ###===============================================================
