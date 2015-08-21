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

def DeprojectGaussian(imax2d, sig2d, q2d, pa, **kwargs) :
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
    imax2d : float
          Maximum amplitude of the Gaussian
    sig2d : float
          Sigma (dispersion) of the Gaussian
    q2d : float
          Axis ratio of the Gaussian
    pa : float
          Position Angle of the Gaussian (starting from North=Vertical up axis,
          going positive counter-clockwise). In degrees.

    Optional Parameters
    -------------------
    euler_angles: array or tuple of 3 floats
          Viewing Angles (Phi, Theta, Psi). All in degrees. 
          Default is [0.,90.,0.] (edge-on).
    inclination : float
          Inclination of the system. This will be the priority on euler_angles.
    geometry : string. Can be 'oblate', 'prolate' or 'triaxial'. 
          Imposing the geometry on the deprojection. Will thus check if this is 
          compatible with the input parameters.
    qzx : float
          In certain cases (e.g. face -on)
          you need to specify an axis ratio here.
          for sigmaZ / sigmaX
    qzy : float
          In certain cases (e.g., phi=Pi/2)
          you need to specify an axis ratio here.
          for sigmaZ / sigmaY

    Return
    ------
    imax3d : float
          Maximum amplitude of the 3D Gaussian
    sig3d : float
          Sigma (dispersion) of the 3D Gaussian
    qzx : float
          Axis ratio (Z/X) of the 3D Gaussian sigmas
    qzy : float
          Axis ratio (Z/Y) of the 3D Gaussian sigmas
    pa : float
          Position Angle of the Gaussian (starting from North=Vertical up axis,
          going positive counter-clockwise). In degrees.
    euler_angles: array or tuple of 3 floats for triaxial case
          For prolate and oblate cases, one euler_angle (the inclination)
          is needed.
          Viewing Angles (Phi, Theta, Psi). All in degrees. 
          Default is [0.,90.,0.] (edge-on).
    """

    # Verbose or not
    verbose = kwargs.get("verbose", False)

    # Defining the error output if something goes wrong
    error_output = 0., 0., 0., 0., 0., 0., False

    # Check which geometry we wish
    geometries = ['oblate', 'prolate', 'triaxial']
    geometry= kwargs.get("geometry", 'oblate')
    if geometry not in geometries:
        print("ERROR: you should choose a geometry in the following list: ", 
                geometries)
        return error_output

    # Euler Angles transformed in radian and checked
    inclination = np.deg2rad(np.float(kwargs.get('inclination', default_inclination)))
    if 'inclination' in kwargs :
        if verbose:
            print("Found an inclination. Other Euler angles will be ignored")
        euler_angles = [0., inclination, 0.]
    elif 'euler_angles' in kwargs : 
        euler_angles = np.deg2rad(np.float(kwargs.get('euler_angles', 
                                   [0., default_inclination, 0.,])))
        if size(euler_angles) != 3 :
            print("ERROR: euler_angles should always" 
                    "have 3 values (tuple or array)")
            return error_output
        inclination = euler_angles[1]
    else :
        if verbose :
            print("WARNING: no set inclination or euler_angles.")
            print("      Using a default deprojection for the edge-on case.")
            print("      Inclination = 90 degrees.")
        euler_angles = [0., inclination, 0.]

    # transferring into the angles names
    [phi, theta, psi] = euler_angles

    # Optional parameters in case this is useful
    # By default, set to 1.
    qzx = kwargs.get("qzx", 1.0)
    qzy = kwargs.get("qzy", 1.0)

    if verbose :
        print("Geometry will be ", geometry)

    # OBLATE Geometry --------------------------------------------------------
    if geometry == 'oblate' :
        if psi != 0.  or phi != 0 :
            if verbose :
                print("WARNING: second and third euler angle"
                        " - Phi and Psi - will be ignored")
        #-------------------------------------------
        # FACE-ON case
        if theta == 0. :
            if q2d != 1 :
                print("ERROR: cannot deproject this model"
                    "as component %d does not have axis ratio of 1!")
            if verbose:
                print("WARNING: this is a face-on case"
                        "An arbitrary value for qzx")
        #-------------------------------------------
        # EDGE-ON case
        elif theta == np.pi/2. :
            if verbose :
                print "Edge-on deprojection\n"
            qzx = q2d
        #-------------------------------------------
        else :
            cosi2 = cos(theta) * cos(theta)
            sini2 = sin(theta) * sin(theta)
            if cosi2 > q2d**2 :
                if verbose :
                    maxangle = np.rad2deg(np.arccos(q2d))
                    print("ERROR: cannot deproject this Gaussian."
                            " Max angle is %f Degrees" %(maxangle))
                    return error_output
            qzx = np.sqrt((q2d**2 - cosi2) / sini2)

        qzy = qzx
        sig3d = sig2d
        imax3d = imax2d *  q2d / (np.sqrt(2. * np.pi) * qzx * sig2d)
        
    elif prolate :
        pass
    elif triaxial :
        pass

    return imax3d, sig3d, qzx, qzy, pa, np.rad2deg(euler_angles)
