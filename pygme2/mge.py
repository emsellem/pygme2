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

try:
    import astropy as apy
    from astropy.modeling import models as astropy_models
    from astropy.modeling import Parameter, Model, Fittable1DModel
    from astropy import constants as constants, units as units
except ImportError:
    raise Exception("astropy is required for this module")

__version__ = '0.0.1 (14 August 2014)'

class MGEModel(BaseModel) :
    """ MGE model 

    This class defines the basic MGE model, which should include both
    a reference 2D Base model made of n_gaussians Gaussians, and the
    associated 3D Gaussians, using the viewing Euler Angles
    """

    def __init__(self, **kwargs) :
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
        BaseModel.__init__(self, **kwargs)

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

