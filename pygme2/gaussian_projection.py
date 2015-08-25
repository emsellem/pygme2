# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Gaussian deprojection and projection module

This module takes care of projecting a 3D Gaussian onto a 2D one
or deproject a 2D Gaussian in a 2D Gaussian.

This involved defining the Euler angles (viewing angles).

For questions, please contact Eric Emsellem at eric.emsellem@eso.org
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2015, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# Importing Modules
import numpy as np
from numpy import cos, sin, tan, arccos, arcsin

__version__ = '0.0.1 (14 August 2014)'
# ==========================================================
# Version 0.0.1: EE - Rewriting
# ==========================================================
# This is a new version of an older package
# This is inspired from the C programme (EE - 1991/1992) 
# and the python version from an earlier pygme package version
# -----------------------------------------------------------

# Default inclination in degrees
default_inclination = 90.0

def _check_consistency_size(list_arrays) :
    if len(list_arrays) == 0 :
        return True

    return all(myarray.size == list_arrays[0].size for myarray in list_arrays)

def _rotation_matrix(angle, axis='z') :
    """Rotation matrix
    Using an angle (in radians), provides the matrix for rotation
    around the axis defined in axis ('x', 'y', or 'z')

    Parameters
    ----------
    angle : float
            Angle in radians
    axis : float
           'x', 'y' or 'z' (default is 'z'). Axis around which to rotate
    """
    cosangle = cos(angle)
    sinangle = sin(angle)
    if axis == 'z' :
        return np.array(
                [[cosangle, -sinangle, 0],
                 [sinangle, cosangle,  0],
                 [0,        0,         1]])
    elif axis == 'y' :
        return np.array(
                 [[cosangle,  0, sinangle],
                  [0,         1,        0],
                  [-sinangle, 0, cosangle]])
    elif axis == 'x' :
        return np.array(
                 [[1,        0,         0],
                  [0, cosangle, -sinangle],
                  [0, sinangle,  cosangle]])
    else : return np.eye(3, dtype=np.float32)

def _check_viewing_angles(euler_angles=None, inclination=None, verbose=False) :
    """Gather the arguments for deprojection

    This includes:

    Parameters
    ----------
    euler_angles : set of 3 floats
    inclination : float

    Return
    ------
    euler_angle : array of 3 floats
                  The 3 Euler angles in degrees
    """
    # Euler Angles transformed in radian and checked
    if inclination is not None:
        if verbose:
            print("Found an inclination. Other Euler angles will be ignored")
        return np.array([0., inclination, 0.])
    elif euler_angles is not None :
        if np.atleast_1d(np.array(euler_angles)).size != 3 :
            if verbose :
                print("ERROR: euler_angles should always" 
                        "have 3 values (tuple or array)")
            return np.array([0., default_inclination, 0.])
        else : return np.array(euler_angles)
    else :
        if verbose :
            print("WARNING: no set inclination or euler_angles.")
            print("      Using a default deprojection for the edge-on case.")
            print("      Inclination = 90 degrees.")
        return np.array([0., default_inclination, 0.])

def _check_3D_axisratios(qzx=None, qzy=None, qxy=None, verbose=False) :
    """Gather the arguments for deprojection

    This includes:

    Parameters
    ----------
    qzx : float
    qzy : float
    qxy : float
          Axis ratios, sometimes needed to impose values

    Return
    ------
    """
    # Having all axis ratios either from given input
    # or by default, set to 1.
    if qzx <= 0. :
        # Forcing qzx to 1.
        qzx = 1.
        if verbose :
            print("WARNING: input qzx is <= 0, forcing it to 1")
    if qzy <= 0. :
        # Forcing qzx to 1.
        if verbose :
            print("WARNING: input qzy is <= 0, forcing it to 1")
        qzy = 1.
    if qxy <= 0. :
        # Forcing qzx to 1.
        qxy = 1.
        if verbose :
            print("WARNING: input qxy is <= 0, forcing it to 1")

    if (qzx is not None) :
        if (qzy is not None) :
            qxy = qzy / qzx
        if (qxy is not None) :
            qzy = qzx * qxy
    else :
        if (qzy is not None) :
            if (qxy is not None) :
                qzx = qzy / qxy
            else :
                qzx = qzy
                qxy = 1.
        else :
            if (qxy is not None) :
                qzx = 1. / qxy
                qzy = 1.
            else :
                return 1., 1., 1.
    return qzx, qzy, qxy

def DeprojectGaussian2D(Gaussian2D, geometry='oblate', euler_angles=None, inclination=None, 
        qzx=None, qzy=None, qxy=None, verbose=False) :
    """Deprojection module to deproject 2D Gaussians in 3D.
    This used the Euler angles (viewing angles) as well as the observed 
    position angle (PA) of the 2D Gaussians.

    The 3D Gaussian is defined by a original [x,y,z] reference frame.
    It it thus rotated via 4 angles and axes:
        1- Phi around the Oz axis
        2- Theta around the Phi(Ox) axis
        3- Psi around the Theta(Phi(Oz)) axis
        4- and finally the PA which a rotation around the Theta(Phi(Oz)) axis

    Hence if we define:
        [x, y, z] as the original axes
        [x1, y1, z1] as the Phi rotated axes
        [x2, y2, z2] as the Theta+Phi rotated axes
        [x', y', z'] as the observed axes 

        Then the rotations are:
        1- Phi around the Oz axis
        2- Theta around the Ox1 axis
        3- Psi around the Oz2 axis
        4- PA around the Oz2 axis (since rotation by Psi does not affect Oz2)

    Parameters
    ----------
    Gaussian2D: MGEGaussian2D
          The class should have the parameters: imax2d, sig2d, q2d and pa, 
          namely the amplitude, sigma, axis ratio and 
          position Angle of the Gaussian (starting from North=Vertical up axis,
          going positive counter-clockwise). In degrees.

    Optional Parameters
    -------------------
    geometry : string. Can be 'oblate', 'prolate' or 'triaxial'. 
          Imposing the geometry on the deprojection. Will thus check if this is 
          compatible with the input parameters.
          Default is 'oblate'
    euler_angles: array or tuple of 3 floats
          Viewing Angles (Phi, Theta, Psi). All in degrees. 
          Default is [0.,90.,0.] (edge-on).
    inclination : float
          Inclination of the system. This will be the priority on euler_angles.
          Default is 90.0 degrees (edge-on)
    qzx : float
          In certain cases (e.g. face -on)
          you need to specify an axis ratio here.
          for sigmaZ / sigmaX. Default would go to 1.
    qzy : float
          In certain cases (e.g., phi=Pi/2)
          you need to specify an axis ratio here.
          for sigmaZ / sigmaY. Default would go to 1.
    qxy : float
          In certain cases (e.g., phi=Pi/2)
          you need to specify an axis ratio here.
          for sigmaZ / sigmaY. Default would go to 1.

    verbose : Boolean. 
          Default to False

    Return
    ------
    Gaussian3D : MGEGaussian3D structure
          which includes imax3d, sig3d, qzx, qzy, psi, theta, phi
          Namely the amplitude, sigma, axis ratios and viewing angles
    """
    from BaseMGE import MGEGaussian3D

    # Transferring the parameters for legibility
    imax2d = Gaussian2D.imax2d
    sig2d = Gaussian2D.sig2d
    q2d = Gaussian2D.q2d
    pa = Gaussian2D.pa

    # Defining the error output if something goes wrong
    error_output = 0., 0., 0., 0., 0., 0., False

    # Check which geometry we wish
    geometries = ['oblate', 'prolate', 'triaxial']
    if geometry not in geometries:
        if verbose :
            print("ERROR: you should choose a geometry in the following list: %s"%geometries)
        return error_output

    # Reading input arguments
    euler_angles = _check_viewing_angles(euler_angles, inclination, verbose)
    qzx, qzy, qxy = _check_3D_axisratios(qzx, qzy, qxy, verbose)

    # transferring into the angles names and in radians
    [phi, theta, psi] = np.deg2rad(euler_angles)
    pa_rad = np.deg2rad(pa)

    # ========================================================================
    # Starting to look at the present case and starting the calculations
    # ========================================================================
    if theta == 0. :
        if (geometry == 'oblate') & (qzx >= 1) :
            if verbose :
                print("ERROR: for a face-on oblate model,"
                      " qzx should be defined and always <= 1")
            return error_output
        elif (geometry == 'prolate') & (qzx <= 1)  :
            if verbose :
                print("ERROR: for a face-on prolate model, "
                      " qzx should be defined and always >= 1")
            return error_output

    if verbose :
        print("Geometry for the deprojection is ", geometry)

    # OBLATE Geometry --------------------------------------------------------
    if geometry == 'oblate' :
        if psi != 0.  or phi != 0 :
            if verbose :
                print("WARNING: second and third euler angle"
                        " - Phi and Psi - will be ignored")
        #-------------------------------------------
        if q2d > 1 : sig2d, q2d, pa_rad = _rotate_pa(sig2d, q2d, pa_rad)

        imax3d, sig3d, qzx, qzy = _axisymmetric_deproj(imax2d, sig2d, 
                                                 q2d, theta, verbose)

    # PROLATE Geometry --------------------------------------------------------
    elif  geometry == 'prolate' :
        if psi != np.pi / 2. or phi != 0 :
            if verbose :
                print("WARNING: second and third euler angle"
                        " - Phi and Psi - will be ignored")
        if theta == 0. :
            phi = -psi
        #-------------------------------------------
        if q2d < 1 : sig2d, q2d, pa_rad = _rotate_pa(sig2d, q2d, pa_rad)
        imax3d, sig3d, qzx, qzy = _axisymmetric_deproj(imax2d, sig2d, 
                                                 q2d, theta, verbose)
         
    # TRIAXIAL Geometry --------------------------------------------------------
    elif  geometry == 'triaxial' :
        # Here are the special cases which must be dealt with independently
        # These correspond to cases when the line-of-sight is one
        # of the principal axis, leading to degeneracies.
        #
        # First the FACE-ON case
        if theta == 0. :
            if (psi != -psi) & verbose :
                print("ERROR: for a triaxial model, when theta=0"
                        "(face-on model), psi must be equal to psi"
                        "Please change your viewing angles")
            if verbose:
                print("WARNING: this is a face-on case"
                        "An arbitrary value for qzy (< 1) needs to be used")
            qzx = q2d * qzy
            sig3d = np.sqrt(q2d) * sig2d

        # Then the EDGE-ON CASE
        elif theta == np.pi / 2. :
            if psi != 0 & verbose:
                print("ERROR: for a triaxial model, when theta=90.0"
                        "(edge-on model), psi must be equal to 0"
                        "Please change your viewing angles")
            if verbose:
                print("WARNING: this is an edge-on triaxial case"
                        "An arbitrary value for qxy needs to be used")
            qzx = q2d * np.sqrt(sin(phi)**2 / qxy**2 + cos(phi)**2)
            sig3d = q2d * sig2d / np.sqrt(qzx * qzy)

        # Here we have the other phi=0 or pi/2 cases
        elif phi == 0. :
            qzx = q2d * qzy / np.sqrt(qzy**2 * sin(theta)**2 + cos(theta)**2)
            sig3d = sig2d / np.sqrt(qxy)
        elif phi == np.pi / 2. :
            qzx = q2d * qzx / np.sqrt(qzx**2 * sin(theta)**2 + cos(theta)**2)
            sig3d = sig2d * np.sqrt(qxy)
        
        # Now the general case
        else :
            cospsi = cos(psi)
            sinpsi = sin(psi)
            cospsi2 = cospsi**2
            sinpsi2 = sinpsi**2
            q2d2 = q2**2
            a = sinpsi2 + q2d2 * cospsi2
            b = -(1. - q2d2) * sipsi * cospsi
            c = q2d2 * sinpsi2 + cospsi2
            c2theta = c - 2* b / (tan(2.* phi) / cos(theta))
            lbda = (a - c2theta  * cos(theta)**2) /  \
                     (c * c2 * sin(theta)**2 - b**2 * tan(theta)**2)
            if lbda < 0 :
                if verbose :
                    print("WARNING: this triaxial deprojection has no solution"
                            " because lambda is negative - see Monnet et al. 1992"
                            " Please Enter new viewing angles")
                return error_output
            j = lbda * a
            k = lbda * b
            l = lbda * c
            qzx = np.sqrt(l - k / (tan(phi) * cos(theta)))
            qzy = np.sqrt(l + k * tan(phi) / cos(theta))
            sig3d = np.sqrt(qzx * qzy / lbda) * sig2d

	    imax3d = imax2d * q2d * sig2d \
                / (np.sqrt(2. * np.pi * qzx * qzy) * sig3d)

    return MGEGaussian3D(imax3d, sig3d, qzx, qzy, psi, theta, phi)


def _rotate_pa(sig2d, q2d, pa) :
    """Rotate the pa by Pi / 2.
    This means also changing sigma and the axis ratio q2d

    Parameters
    ----------
    sig2d : float
        Sigma
    q2d : float
        Axis ratio
    pa : float
        Position angle in radians

    Return
    ------
    sig2d : float
    q2d : float
    pa : float
         Sigma, axis ratio and pa (radians)
         after the rotation (qout = 1 / qin, sigout = sigin * qin)
    """
    sig2d = sig2d * q2d
    q2d = 1. / q2d
    pa = pa + np.pi / 2.
    return sig2d, q2d, pa

def _axisymmetric_deproj(imax2d, sig2d, q2d, inclination, verbose) :
    """Deprojection in the axisymmetric case (oblate/prolate)
    It uses an inclination and the input parameters

    Parameters
    ----------
    imax2d : float
    sig2d : float
    q2d : float
         Amplitude, sigma, and axis ratio of the 2D Gaussian
    inclination: float
         Inclination in radians
    verbose: Boolean
         Whether or not to print some output and comments

    Return
    ------
    imax3d : float
    sig3d : float
    qxz : float
    qyz : float
         Amplitude, sigma and axis ratio(s) of the 3D deprojected
         Gaussian.
    """
    # Defining the error output if something goes wrong
    error_output = 0., 0., 0., 0., 0., 0., False

    # FACE-ON case
    if inclination == 0. :
        if q2d != 1 & verbose:
            print("ERROR: cannot deproject this model"
                "as component %d does not have axis ratio of 1!")
        if verbose:
            print("WARNING: this is a face-on case"
                    "An arbitrary value for qzx needs to be used")
    #-------------------------------------------
    # EDGE-ON case
    elif inclination == np.pi/2. :
        if verbose :
            print "Edge-on deprojection\n"
        qzx = q2d
    #-------------------------------------------
    else :
        cosi2 = cos(inclination) * cos(inclination)
        sini2 = sin(inclination) * sin(inclination)
        if cosi2 > q2d**2 :
            if verbose :
                maxangle = np.rad2deg(arccos(q2d))
                print("ERROR: cannot deproject this Gaussian."
                        " Max angle is %f Degrees" %(maxangle))
                return error_output
        qzx = np.sqrt((q2d**2 - cosi2) / sini2)

    qzy = qzx
    sig3d = sig2d
    imax3d = imax2d * q2d / (np.sqrt(2. * np.pi) * qzx * sig2d)

    return imax3d, sig3d, qzx, qzy

