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
#from rwcfor import floatMGE
#from mge_miscfunctions import print_msg
import os

from BaseMGE import BaseMGEModel, BaseGaussian2D, BaseGaussian3D, BaseMultiGaussian2D, BaseMultiGaussian3D


try:
    import astropy as apy
    from astropy.modeling import models as astropy_models
    from astropy.modeling import Parameter, Model, Fittable1DModel
    from astropy import constants as constants, units as units
except ImportError:
    raise Exception("astropy is required for this module")

__version__ = '0.0.1 (14 August 2014)'

class MGEModel(BaseMGEModel) :
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

        # Truncation Method = Default is Ellipsoid, (can also be Cylindre)
        self.truncation_method = kwargs.get("truncation_method", "Ellipsoid")   
        # Default Truncation radius in parsec
        self.mcut = kwargs.get("mcut", 50000.)

        # Initial value for the Gravitational Constant
        # G is in (km/s)2. Msun-1 . pc .
        self.GGRAV = constants.G.to(units.km**2 * units.pc 
                         / units.s**2 / units.M_sun).value
        # Distance of the model in parsec
        self.distance = kwargs.get("distance", 1.0e6)

        # Viewing angles in degrees
        # Second value is inclination for axisymmetric systems
        # 90 degrees inclination means edge-on view
        self.euler_angles = kwargs.get("euler_angles", np.array([0., 90., 0.]))

        # Input Parameters
        self.n_gaussians = np.int(kwargs.get("n_Gaussians", 1))
        
        # Now setting the 2D / 3D Gaussians
        BaseMGEModel.__init__(self, **kwargs)
        
        self.init_BasePhotModel()
        
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
    
    def _reset(self) :
        self.n_gaussians = 0
        self.euler_angles[0] = self.euler_angles[1] = self.euler_angles[2] = 0
        self.model2d = None
        self.model3d = None

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
        mgeout.write("## ID                  Imax    Sigma       QxZ       QyZ   \n")
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

def create_mge(outfilename=None, overwrite=False, outdir=None, **kwargs) :
    """Create an MGE ascii file corresponding to the input parameters
    """
    
    tmpMGE = MGEModel(**kwargs)
    
    tmpMGE.write_mge(outfilename,overwrite,outdir)
        
    ###===============================================================
