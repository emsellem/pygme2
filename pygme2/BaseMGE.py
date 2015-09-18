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

from gaussian_projection import _rotation_matrix
from gaussian_projection import _check_viewing_angles
from gaussian_projection import _check_3D_axisratios
from gaussian_projection import _check_consistency_size

__version__ = '0.0.2 (25 August 2014)'
#__version__ = '0.0.1 (14 August 2014)'
# ==========================================================
# Version 0.0.1: EE - First drafted structure
# Version 0.0.2: EE - Converging on the structure
# ==========================================================
# This is a full rewriting of an old package
# The restructuring was made to take advantage of the model
# structure from astropy.
#
# -----------------------------------------------------------

def update_properties(func):
    print "Updating..."
    

class MyList(list):
    def __init__(self,type):
        self.type = type
    
    @update_properties
    def pop(self, index):
        self.pop(index)
    

# ========================================
# Gaussian 1D model
# Inherited from astropy
# ----------------------------------------
class BaseGaussian1D(Model) :
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
                amplitude=imax1d, stddev=sig1d, mean=xcentre1d)

class BaseMultiGaussian1D(object) :
    """Base Multi 1D Gaussian 

    Set of n_gaussians 1D Gaussians

    Parameters
    ----------

    n_gaussians : int
                number of Gaussians
    imax1d: array of floats
    sig1d: array of floats
         Amplitudes, sigmas of the 1D Gaussians

    xcentre1d : float
          Centre of the Gaussians in observed coordinates

    Return 
    ------
    A Base 1D model class with n_gaussians 1D Gaussians
    This is a composite model from BaseGaussian1D.
    """

    def __init__(self, imax1d, sig1d, **kwargs) :
        """Initialise the set of 1D Gaussians
        """

        # All the following parameters have getters/setters
        # As they are set to be views of the model3d.parameters array
        imax1d = np.atleast_1d(np.asarray(imax1d, dtype=np.float32))

        n_gaussians = np.size(imax1d)
        sig2d =  np.atleast_1d(np.asarray(sig1d, dtype=np.float32))

#        if any(sig1d == 0) :
#            print("ERROR: sigma's should be non-zeros")

#        if not _check_consistency_size([imax1d, sig1d]) :
#            print("ERROR: not all input arrays (imax, sigma)"
#                  " have the same size")

        # All the following parameters have getters/setters
        # As they are set to be views of the model2d.parameters array
        xcentre1d = _read_resize_arg("xcentre1d", n_gaussians, 0., **kwargs)

        self.model1d =  None
        for i in range(n_gaussians) :
            newmodel1d = BaseGaussian1D(imax1d[i],
                    sig1d[i], xcentre1d[i])
            if self.model1d is None :
                self.model1d = [newmodel1d]
            else :
                self.model1d.append(newmodel1d)

    def evaluate(self, x) :
        G1 = np.zeros(np.size(x))
        for i in range(np.size(self.model1d)) :
            G1 += self.model1d[i](x)
        return G1
    
    def RemoveGaussian(self, index):
        self.model1d.pop(index)


# ========================================
# Gaussian 2D model
# Inherited from astropy
# ----------------------------------------
class BaseGaussian2D(Model) :
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

#    def evaluate(x, y, imax2d, sig2d, q2d, pa, xcentre2d=0.0, ycentre2d=0.0) :
#        pa_x = np.deg2rad(pa + 90.0)
#        cost2 = np.cos(pa_x) ** 2
#        sint2 = np.sin(pa_x) ** 2
#        sin2t = np.sin(2. * pa_x)
#        xstd2 = sig2d ** 2
#        ystd2 = (sig2d * q2d) ** 2
#        xdiff = x - xcentre2d
#        ydiff = y - ycentre2d
#        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
#        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
#        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
#        return amplitude * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
#                                    (c * ydiff ** 2)))    
    @staticmethod
    def evaluate(x, y, imax2d, sig2d, q2d, pa, xcentre2d, ycentre2d) :
        return astropy_models.Gaussian2D.evaluate(x, y, 
                amplitude=imax2d, x_stddev=sig2d, y_stddev=sig2d * q2d, 
                theta=np.deg2rad(pa+90.0), x_mean=xcentre2d, y_mean=ycentre2d)


class BaseMultiGaussian2D(object) :
    """Base 2D Model for MGE

    Set of n_gaussians 2D Gaussians

    Parameters
    ----------

    n_gaussians : int
                number of Gaussians
    imax2d: array of floats
    sig2d: array of floats
    q2d: array of floats
         Amplitudes, sigmas and axis ratios of the 2D Gaussians

    pa : float
        Position angle of the Gaussians

    xcentre2d : float
    ycentre2d : float
          Centre of the Gaussians in observed coordinates

    Return 
    ------
    A Base 2D model class with n_gaussians 2D Gaussians
    This is a composite model from BaseGaussian2D.
    """

    def __init__(self, imax2d, sig2d, q2d, **kwargs) :
        """Initialise the set of 2D Gaussians
        """

        # All the following parameters have getters/setters
        # As they are set to be views of the model3d.parameters array
        imax2d = np.atleast_1d(np.asarray(imax2d, dtype=np.float32))

        n_gaussians = np.size(imax2d)
        sig2d =  np.atleast_1d(np.asarray(sig2d, dtype=np.float32))
        q2d =  np.atleast_1d(np.asarray(q2d, dtype=np.float32))

        if any(q2d == 0) or any(sig2d == 0) :
            print("ERROR: sigma's and q's should be non-zeros")

        if not _check_consistency_size([imax2d, sig2d, q2d]) :
            print("ERROR: not all input arrays (imax, sigma, q)"
                  " have the same size")

        # All the following parameters have getters/setters
        # As they are set to be views of the model2d.parameters array
        pa = _read_resize_arg("pa", n_gaussians, -90., **kwargs)
        xcentre2d = _read_resize_arg("xcentre2d", n_gaussians, 0., **kwargs)
        ycentre2d = _read_resize_arg("ycentre2d", n_gaussians, 0., **kwargs)

        self.model2d =  None
        for i in range(n_gaussians) :
            newmodel2d = BaseGaussian2D(imax2d[i],
                    sig2d[i], q2d[i], pa[i], xcentre2d[i], ycentre2d[i])
            if self.model2d is None :
                self.model2d = [newmodel2d]
            else :
                self.model2d.append(newmodel2d)

    def evaluate(self, x, y) :
        G2 = np.zeros(np.size(x))
        for i in range(np.size(self.model2d)) :
            G2 += self.model2d[i](x,y)
        return G2
    
    def RemoveGaussian(self, index):
        self.model2d.pop(index)
        
    def AddGaussian(self, item):
        self.model2d.append(item)

    def deproject(self, **kwargs) :
        """
        Parameters
        ----------
        geometry : string
                 One of the three 'oblate', 'prolate' or 'triaxial'
                 Default is 'oblate'.
        euler_angles: array of 3 floats
                 Three viewing angles (in degrees). Default is [0, 90, 0].
        Return
        ------
        model3d : a Base3DModel, including the 3D Gaussians
        """
        from gaussian_projection import DeprojectGaussian2D

        model3d = None
        for model2d in self.model2d :
            newmodel3d = DeprojectGaussian2D(model2d, **kwargs)
            if model3d is None :
                model3d = [newmodel3d]
            else :
                model3d.append(newmodel3d)

        return model3d


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
    phi : float
    theta : float
    psi : float
          Euler angles (in radians)

    Return
    ------
    An astropy.modeling compatible class for a 3D Gaussian
    """
    amplitude = Parameter()
    x_mean = Parameter()
    y_mean = Parameter()
    z_mean = Parameter()
    x_stddev = Parameter()
    y_stddev = Parameter()
    z_stddev = Parameter()
    psi = Parameter()
    theta = Parameter()
    phi = Parameter()

    def __init__(self, amplitude, x_mean=0.0, y_mean=0.0, z_mean=0.0, 
            x_stddev=1.0, y_stddev=1.0, z_stddev=1.0, 
            psi=0.0, theta=np.pi/2., phi=0.0, **kwargs):

        self.x_mean = x_mean
        self.y_mean = y_mean
        self.z_mean = z_mean
        self.x_sttdev = x_stddev
        self.y_sttdev = y_stddev
        self.z_sttdev = z_stddev
        self.psi = psi
        self.theta = theta
        self.phi = phi

    @staticmethod
    def evaluate(x, y, z, amplitude, x_mean, y_mean, z_mean, 
                 x_stddev, y_stddev, z_stddev, psi, theta, phi) :
        """Three dimensional Gaussian function"""

        # Observed Position Angle of each Gaussian is PA
        # The angle for the deprojection is Psi.
        # Hence to have deprojected aligned axes after deprojection
        # we need to have Psi = PA + cte
        # In any cases, the evaluation here is done for each Gaussian
        # with its own Psi angle
        # Rotation is done first with Psi, then Theta, then Phi

        # First convert into radians
        [psi_rad, theta_rad, phi_rad] = np.deg2rad([psi, theta, psi])
        # Then compute the total rotation matrix
        M_euler = []
        M_euler.append(_rotation_matrix(psi_rad, 'z'))
        M_euler.append(_rotation_matrix(theta_rad, 'x'))
        M_euler.append(_rotation_matrix(phi_rad, 'z'))
        M_rot = reduce(np.dot, M_euler[::-1])
        # Then apply it on the coordinates
        [xr, yr, zr] = M_rot.dot([x - x_mean, y - y_mean, z - z_mean])

        return amplitude * np.exp(-0.5 * ((xr / x_stddev)**2 + 
                    (yr / y_stddev)**2 + (zr / z_stddev)**2))


# ========================================
# Gaussian 3D model
# Inherited from self-built astropy model
# ----------------------------------------
class BaseGaussian3D(Model) :
    """ MGE 3D Gaussian model class using astropy

    Parameters
    ----------
    **kwargs : kwargs
         Set of free arguments given to MGE_Gaussian2D

    Input
    -----
    imax3d : float
    sig3d : float
    qzx : float
    qzy : float
          Input amplitude, sigma and axis ratios
    phi : float
    theta : float
    psi : float
          Input Euler angles (viewing angles)
          This corresponds to axes X, Y, and X
          for the principal axes of the 3D Gaussians
          to be transformed via rotations into X', Y', Z':
             1- around the Oz axis (Psi)
             2- around the Ox axis (Theta)
             3- around the Oz axis (Phi)
    xcentre3d : float
    ycentre3d : float
    zcentre3d : float
          Centre of the Gaussians in observed coordinates

    Return
    ------
    A 3D MGE Gaussian model class
    """
    inputs = ("x", "y", "z")
    outputs = ("G3D",)

    imax3d = Parameter(default=1.0)
    sig3d = Parameter(default=1.0)
    qzx = Parameter(default=1.0)
    qzy = Parameter(default=1.0)
    psi = Parameter(default=0.)
    theta = Parameter(default=90.)
    phi = Parameter(default=0.)
    xcentre3d = Parameter(default=0.0)
    ycentre3d = Parameter(default=0.0)
    zcentre3d = Parameter(default=0.0)

    @staticmethod
    def evaluate(x, y, z, imax3d, sig3d, qzx, qzy, 
            psi, theta, phi, xcentre3d, ycentre3d, zcentre3d) :
        return astropy_models.Gaussian3D.evaluate(x, y, z, amplitude=imax3d, 
                x_stddev=sig3d, y_stddev=sig3d * qzx / qzy, z_stddev=sig3d * qzx, 
                psi=psi, theta=theta, phi=phi,
                x_mean=xcentre3d, y_mean=ycentre3d, z_mean=zcentre3d)


class BaseMultiGaussian3D(object) :
    """Base 3D Model for MGE

    Set of n_gaussians 3D Gaussians

    Parameters
    ----------

    n_gaussians : int
                number of Gaussians
    imax3d: array of floats
    sig3d: array of floats
    qzx: array of floats
    qzy: array of floats
         Amplitudes, sigmas and axis ratios of the 3D Gaussians

    psi : float
    theta : float
    phi : float
        Euler angles (in degrees) of the 3D Gaussians

    xcentre3d : float
    ycentre3d : float
    zcentre3d : float
          Centre of the Gaussians in observed coordinates

    Return 
    ------
    A Base 3D model class with n_gaussians 3D Gaussians
    This is a composite model from MGEGaussian3D.
    """
    def __init__(self, imax3d, sig3d, qzx, qzy, **kwargs) :
        """Initialise the set of 3D Gaussians
        """

        # All the following parameters have getters/setters
        # As they are set to be views of the model3d.parameters array
        imax3d = np.atleast_1d(np.ndarray(imax3d, dtype=np.float32))

        n_gaussians = np.size(imax3d)
        sig3d = np.atleast_1d(np.ndarray(sig3d, dtype=np.float32))
        qzx = np.atleast_1d(np.ndarray(qzx, dtype=np.float32))
        qzy = np.atleast_1d(np.ndarray(qzy, dtype=np.float32))

        if any(qzx == 0) or any(qzy == 0) or any(sig3d == 0) :
            print("ERROR: sigma's and q's should be non-zeros")

        if not check_consistency_size([imax3d, sig3d, qzx, qzy]) :
            print("ERROR: not all input arrays (imax, sigma, q)"
                  " have the same size")

        psi = _read_resize_arg(kwargs, "psi", n_gaussians, 0.)
        theta = _read_resize_arg(kwargs, "theta", n_gaussians, 90.0)
        phi = _read_resize_arg(kwargs, "phi", n_gaussians, 0.)
        xcentre3d = _read_resize_arg(kwargs, "xcentre3d", n_gaussians, 0.)
        ycentre3d = _read_resize_arg(kwargs, "ycentre3d", n_gaussians, 0.)
        zcentre3d = _read_resize_arg(kwargs, "zcentre3d", n_gaussians, 0.)

        self.model3d = None
        for i in range(n_gaussians) :
            newmodel3d = BaseGaussian3D(imax3d[i],
                           sig3d[i], qzx[i], qzy[i], psi[i], theta[i], phi[i],
                           xcentre3d[i], ycentre3d[i], zcentre3d[i])
            if self.model3d is None :
                self.model3d = [newmodel3d]
            else :
                self.model3d.append(newmodel3d)
        
        self.model3d.n_gaussians = n_gaussians

    def evaluate(self, x, y, z) :
        G2 = np.zeros(np.size(x))
        for i in range(np.size(self.model3d)) :
            G2 += self.model3d[i](x,y,z)
        return G2
    
    def RemoveGaussian(self, index):
        self.model3d.pop(index)
        
    def AddGaussian(self, item):
        self.model2d.append(item)

    def project(self, geometry='oblate', euler_angles=[0, 90., 0.]) :
        pass

def _read_resize_arg(arg, sizearray, defaultvalue, **kwargs) :
    """Read and resize the input arrays
    If the argument is present in the kwargs list, then check if it
    is just 1 number, and in that case resize to an array of sizearray.

    Parameters
    ----------
    **kwargs: list of arguments
    arg: string
         Argument to consider in kwargs
    sizearray : int
         Size of the array
    defaultvalue : float
         Default value to impose if not given

    Return
    ------
    variable: an array with size, sizearray
    """
    variable = kwargs.get(arg, np.zeros(sizearray, dtype=np.float32)+defaultvalue)
    if np.size(variable) == 1 : return np.resize(variable, sizearray)
    else : return variable

class BaseMGEModel(object) :
    """ MGE model 

    This class defines the basic MGE model, which should include both
    a reference 2D Base model made of n_gaussians Gaussians, and the
    associated 3D Gaussians, using the viewing Euler Angles
    """

    def __init__(self, geometry='oblate', euler_angles=None, inclination=None, 
        verbose=False, **kwargs) :
        """Initialise the MGE model

        Set up the geometry and euler_angles, and provide the input 2D model. 
        That model is then deprojected into a 3D MGE model using the input
        geometry and euler_angles.
        """
        from gaussian_projection import _check_viewing_angles, _check_3D_axisratios

        # Input Parameters
        self.geometry = geometry
        self.euler_angles = _check_viewing_angles(euler_angles, inclination, verbose)

        qzx = kwargs.get("qzx", None)
        qzy = kwargs.get("qzy", None)
        qxy = kwargs.get("qxy", None)
        self._input_qzx, self._input_qzy, self._input_qxy = \
                _check_3D_axisratios(qzx, qzy, qxy, verbose) 

        # Getting the 2D model - Gaussians
        self.model2d = BaseMultiGaussian2D(**kwargs).model2d

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
    def xcentre2d(self) :
        """Amplitudes of the 2D Gaussians
        """
        return self.model2d.parameters[4::6]

    @xcentre2d.setter
    def xcentre2d(self, value) :
         self.model2d.parameters[4::6] = value
    # -------------------------------------

    @property 
    def ycentre2d(self) :
        """Amplitudes of the 2D Gaussians
        """
        return self.model2d.parameters[5::6]

    @ycentre2d.setter
    def ycentre2d(self, value) :
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
    def xcentre3d(self) :
        """Amplitudes of the 3D Gaussians
        """
        return self.model3d.parameters[6::9]

    @xcentre3d.setter
    def xcentre3d(self, value) :
         self.model3d.parameters[6::9] = value
    # -------------------------------------
    @property 
    def ycentre3d(self) :
        """Amplitudes of the 3D Gaussians
        """
        return self.model3d.parameters[7::9]

    @ycentre3d.setter
    def ycentre3d(self, value) :
         self.model3d.parameters[7::9] = value
    # -------------------------------------
    @property 
    def zcentre3d(self) :
        """Amplitudes of the 3D Gaussians
        """
        return self.model3d.parameters[8::9]

    @zcentre3d.setter
    def zcentre3d(self, value) :
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
    @property
    def model2d(self) :
        """Model 2D Gaussians
        """
        # Now setting the 2D Gaussians
        return self._model2d

    @model2d.setter
    def model2d(self, value, deproj=True) :
        self._model2d = value
        if (deproj == True) :
            base3d = self._model2d.deproject(self.geometry, self.euler_angles)
            self._model3d = base3d.model3d

    @property
    def model3d(self) :
        """Model 3D Gaussians
        """
        # Now setting the 2D Gaussians
        return self._model3d

    @model3d.setter
    def model3d(self, value, proj=True) :
        self._model3d = value
        if (proj == True) :
            base2d = self._model3d.project(self.geometry, self.euler_angles)
            self._model2d = base2d.model2d
    # --------------------------------------------------

