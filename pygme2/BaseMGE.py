# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Gaussian nD MGE modelling module

This module includes the definitions, via astropy, of the Gaussians 
in 1D, 2D, and 3D.
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
# ==========================================================
# Version 0.0.1: EE - First drafted structure
# ==========================================================
# This is a new version of an old package
# The restructuring was made to take advantage of the model
# structure from astropy.
#
# -----------------------------------------------------------

# ========================================
# Gaussian 1D model
# Inherited from astropy
# ----------------------------------------
class MGEGaussian1D(Model) :
    """ MGE 1D Gaussian model using astropy

    Parameters
    ----------
    **kwargs : kwargs
         Set of free arguments given to MGE_Gaussian1D

    Return
    ------
    MGE_Gaussian1D

    """
    inputs = ("x")
    outputs = ("G1D",)

    imax1d = Parameter(name="imax1d", default=1.0)
    sig1d = Parameter(name="sig1d", default=1.0)
    xcentre1d = Parameter(name="xcentre1d", default=0.0)

    @staticmethod
    def evaluate(x, imax1d, sig1d, xcentre1d) :
        return astropy_models.Gaussian1D.evaluate(x,
                amplitude=imax1d, mean=xcentre1d, stddev=sig1d)


# ========================================
# Gaussian 2D model
# Inherited from astropy
# ----------------------------------------
class MGEGaussian2D(Model) :
    """ MGE 2D Gaussian model using astropy

    Parameters
    ----------
    **kwargs : kwargs
         Set of free arguments given to MGE_Gaussian2D

    Return
    ------
    MGE_Gaussian2D
    """
    inputs = ("x", "y")
    outputs = ("G2D",)

    imax2d = Parameter(default=1.0)
    sig2d = Parameter(default=1.0)
    q2d = Parameter(default=1.0)
    pa = Parameter(default=-90.0)
    xcentre2d = Parameter(default=0.)
    ycentre2d = Parameter(default=0.)

    def evaluate(x, y, imax2d, sig2d, q2d, pa, xcentre2d=0.0, ycentre2d=0.0) :
        theta = np.deg2rad(pa + np.pi / 2.)
        cost2 = np.cos(theta) ** 2
        sint2 = np.sin(theta) ** 2
        sin2t = np.sin(2. * theta)
        xstd2 = sig2d ** 2
        ystd2 = (sig2d * q2d) ** 2
        xdiff = x - xcentre2d
        ydiff = y - ycentre2d
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
        return amplitude * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                                    (c * ydiff ** 2)))    
    @staticmethod
    def evaluate(x, y, imax2d, sig2d, q2d, pa, xcentre2d, ycentre2d) :
        return astropy_models.Gaussian2D.evaluate(x, y, 
                amplitude=imax2d, x_stddev=sig2d, y_stddev=sig2d * q2d, 
                theta=pa, x_mean=xcentre2d, y_mean=ycentre2d)


# ========================================
# Gaussian 3D model built with astropy
# ----------------------------------------
class Gaussian3D(Model) :
    """New Class for the 3D Gaussians

    Class defining the 3D Gaussians (x, y, z)

    Parameters
    ----------
    amplidude : float
                Amplitude of the Gaussian
    x_mean : float
    y_mean : float
    z_mean : float
             Centres for the Gaussians
    x_stddev : float
    y_stddev : float
    z_stddev : float
             Sigma in x, y  and z for the Gaussians
    """
    amplitude = Parameter()
    x_mean = Parameter()
    y_mean = Parameter()
    z_mean = Parameter()
    x_stddev = Parameter()
    y_stddev = Parameter()
    z_stddev = Parameter()
    theta = Parameter()
    phi = Parameter()

    def __init__(self, amplitude, x_mean=0.0, y_mean=0.0, z_mean=0.0, 
            x_stddev=1.0, y_stddev=1.0, z_stddev=1.0, 
            theta=0.0, phi=0.0, **kwargs):

        self.x_mean = x_mean
        self.y_mean = y_mean
        self.z_mean = z_mean
        self.x_sttdev = x_sttdev
        self.y_sttdev = y_sttdev
        self.z_sttdev = z_sttdev
        self.theta = theta
        self.phi = phi

    @staticmethod
    def evaluate(x, y, z, amplitude, x_mean, y_mean, z_mean, 
                 x_stddev, y_stddev, z_stddev, theta, phi) :
        """Three dimensional Gaussian function"""

        xdiff = x - x_mean
        ydiff = y - y_mean
        zdiff = z - z_mean
        return amplitude * np.exp(-0.5 * ((xdiff / x_stddev)**2 + 
                    (ydiff / y_stddev)**2 + (zdiff / z_stddev)**2))

# ========================================
# Gaussian 3D model
# Inherited from self-built astropy model
# ----------------------------------------
class MGEGaussian3D(Model) :
    """ MGE 2D Gaussian model using astropy

    Parameters
    ----------
    **kwargs : kwargs
         Set of free arguments given to MGE_Gaussian2D

    Return
    ------
    MGE_Gaussian2D
    """
    inputs = ("x", "y", "z")
    outputs = ("G3D",)

    imax3d = Parameter(default=1.0)
    sig3d = Parameter(default=1.0)
    qzx = Parameter(default=1.0)
    qzy = Parameter(default=1.0)
    pa = Parameter(default=-90.)
    phi = Parameter(default=0.)
    xcentre3d = Parameter(default=0.0)
    ycentre3d = Parameter(default=0.0)
    zcentre3d = Parameter(default=0.0)

    @staticmethod
    def evaluate(x, y, z, imax3d, sig3d, qzx, qzy, pa, phi, 
            xcentre3d, ycentre3d, zcentre3d) :
        return astropy_models.Gaussian3D.evaluate(x, y, z, amplitude=imax3d, 
                x_mean=xcentre3d, y_mean=ycentre3d, z_mean=zcentre3d,
                x_stddev=sig3d, y_stddev=sig3d * qzx / qzy, z_stddev=sig3d * qzx, 
                theta=pa, phi=phi)


class Base3DModel(object) :
    """Base 3D Model for MGE

    Set of 3D Gaussians
    """
    def __init__(self, *kwargs) :
        """Initialise the set of 3D Gaussians
        """

        # Input Parameters
        n_gaussians = np.int(kwargs.get("n_Gaussians", 1))

        # All the following parameters have getters/setters
        # As they are set to be views of the model3d.parameters array
        imax3d = kwargs.get("imax3d", 
                np.ones(self.n_gaussians, dtype=float32))
        sig3d = kwargs.get("sig3d", np.ones_like(imax3d))
        qzx = kwargs.get("qzx",  np.ones_like(imax3d))
        qzy = kwargs.get("qzy",  np.ones_like(imax3d))
        pa = kwargs.get("pa",  np.ones_like(imax3d))
        phi = kwargs.get("phi",  np.ones_like(imax3d))
        xcentre3d = kwargs.get("xcentre3d",  np.ones_like(imax3d))
        ycentre3d = kwargs.get("ycentre3d",  np.ones_like(imax3d))
        zcentre3d = kwargs.get("zcentre3d",  np.ones_like(imax3d))

        self.model3d = None
        for i in range(n_gaussians) :
            if self.model3d is None :
                self.model3d = MGEGaussian3D(imax3d[i],
                        sig3d[i], qzx[i], qzy[i], pa[i], phi[i],
                        xcentre3d[i], ycentre3d[i], zcentre3d[i])
            else :
                self.model3d += MGEGaussian3D(imax2d[i],
                        sig3d[i], qzx[i], qzy[i], pa[i], phi[i],
                        xcentre3d[i], ycentre3d[i], zcentre3d[i])

        self.model3d.n_gaussians = n_gaussians

class Base2DModel(object) :
    """ Base 2D model for MGE

    Set of 2D Gaussians
    """

    def __init__(self, **kwargs) :
        """Initialise the set of 2D Gaussians
        """

        # Input Parameters
        n_gaussians = np.int(kwargs.get("n_Gaussians", 1))

        # All the following parameters have getters/setters
        # As they are set to be views of the model2d.parameters array
        imax2d = kwargs.get("imax2d", 
                np.ones(self.n_gaussians, dtype=float32))
        sig2d = kwargs.get("sig2d", np.ones_like(imax2d))
        q2d = kwargs.get("q2d",  np.ones_like(imax2d))
        pa = kwargs.get("pa",  np.ones_like(imax2d))
        xcentre2d = kwargs.get("xcentre2d",  np.ones_like(imax2d))
        ycentre2d = kwargs.get("ycentre2d",  np.ones_like(imax2d))

        self.model2d = None
        for i in range(n_gaussians) :
            if self.model2d is None :
                self.model2d = MGEGaussian2D(imax2d[i],
                        sig2d[i], q2d[i], pa[i], xcentre2d[i], ycentre2d[i])
            else :
                self.model2d += MGEGaussian2D(imax2d[i],
                        sig2d[i], q2d[i], pa[i], xcentre2d[i], ycentre2d[i])

        self.model2d.n_gaussians = n_gaussians

class BaseMGEModel(object) :
    """ MGE model 

    This class defines the basic MGE model, which should include both
    a reference 2D Base model made of n_gaussians Gaussians, and the
    associated 3D Gaussians, using the viewing Euler Angles
    """

    def __init__(self, **kwargs) :
        """Initialise the MGE model
        """
        # Input Parameters
        self.n_gaussians = np.int(kwargs.get("n_Gaussians", 1))

        # Now setting the 2D Gaussians
        self.model2d = Base2DModel(**kwargs).model2d

    # =================================================
    #                  2D model parameters
    # Defining the parameters that the User can access
    # imax2d, sig2d, q2d, pa, xcentre, ycentre
    # =================================================
    @property 
    def imax2d(self) :
        """Amplitudes of the 2D Gaussians
        """
        return self.model2d.parameters[::6]

    @imax2d.setter
    def imax2d(self, value) :
         self.model2d.parameters[::6] = value
    # -------------------------------------
    @property 
    def sig2d(self) :
        """Amplitudes of the 2D Gaussians
        """
        return self.model2d.parameters[1::6]

    @sig2d.setter
    def sig2d(self, value) :
         self.model2d.parameters[1::6] = value
    # -------------------------------------

    @property 
    def q2d(self) :
        """Amplitudes of the 2D Gaussians
        """
        return self.model2d.parameters[2::6]

    @q2d.setter
    def q2d(self, value) :
         self.model2d.parameters[2::6] = value
    # -------------------------------------

    @property 
    def pa(self) :
        """Amplitudes of the 2D Gaussians
        """
        return self.model2d.parameters[3::6]

    @pa.setter
    def pa(self, value) :
         self.model2d.parameters[3::6] = value
    # -------------------------------------

    @property 
    def xcentre(self) :
        """Amplitudes of the 2D Gaussians
        """
        return self.model2d.parameters[4::6]

    @xcentre.setter
    def xcentre(self, value) :
         self.model2d.parameters[4::6] = value
    # -------------------------------------

    @property 
    def ycentre(self) :
        """Amplitudes of the 2D Gaussians
        """
        return self.model2d.parameters[5::6]

    @ycentre.setter
    def ycentre(self, value) :
         self.model2d.parameters[5::6] = value
    # --------------------------------------------------
    # =================================================
    #               3D model parameters
    # Defining the parameters that the User can access
    # imax3d, sig3d, qxz, qyz, pa, phi, 
    # xcentre, ycentre, zcentre
    # =================================================
    @property 
    def imax3d(self) :
        """Amplitudes of the 3D Gaussians
        """
        return self.model3d.parameters[::9]

    @imax3d.setter
    def imax3d(self, value) :
         self.model3d.parameters[::9] = value
    # -------------------------------------
    @property 
    def sig3d(self) :
        """Sigma of the 3D Gaussians
        """
        return self.model3d.parameters[1::9]

    @sig3d.setter
    def sig3d(self, value) :
         self.model3d.parameters[1::9] = value
    # -------------------------------------
    @property 
    def qzx(self) :
        """Axis ratio (X/Y) of the 3D Gaussians
        """
        return self.model3d.parameters[2::9]

    @qzx.setter
    def qzx(self, value) :
         self.model3d.parameters[2::9] = value
    # -------------------------------------
    @property 
    def qzy(self) :
        """Axis ratio (X/Z) of the 3D Gaussians
        """
        return self.model3d.parameters[3::9]

    @qzy.setter
    def qzy(self, value) :
         self.model3d.parameters[3::9] = value
    # -------------------------------------
    @property 
    def pa(self) :
        """Position Angles of the 3D Gaussians
        """
        return self.model3d.parameters[4::9]

    @pa.setter
    def pa(self, value) :
         self.model3d.parameters[4::9] = value
    # -------------------------------------
    @property 
    def phi(self) :
        """Angle Phi of the 3D Gaussians
        """
        return self.model3d.parameters[5::9]

    @phi.setter
    def phi(self, value) :
         self.model3d.parameters[5::9] = value
    # -------------------------------------
    @property 
    def xcentre(self) :
        """Amplitudes of the 3D Gaussians
        """
        return self.model3d.parameters[6::9]

    @xcentre.setter
    def xcentre(self, value) :
         self.model3d.parameters[6::9] = value
    # -------------------------------------
    @property 
    def ycentre(self) :
        """Amplitudes of the 3D Gaussians
        """
        return self.model3d.parameters[7::9]

    @ycentre.setter
    def ycentre(self, value) :
         self.model3d.parameters[7::9] = value
    # -------------------------------------
    @property 
    def zcentre(self) :
        """Amplitudes of the 3D Gaussians
        """
        return self.model3d.parameters[8::9]

    @zcentre.setter
    def zcentre(self, value) :
         self.model3d.parameters[8::9] = value
    # -------------------------------------
    @property
    def inclination(self) :
        """Inclination of the model which is just
        the second Euler Angle
        """
        return self.euler_angles[1]

    @inclination.setter
    def inclination(self, value) :
        if value is not None :
            self.euler_angles[1] = value
    # --------------------------------------------------

