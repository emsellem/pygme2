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
    qxz : float
          In certain cases (e.g. face -on)
          you need to specify an axis ratio here.
    qyz : float
          In certain cases (e.g., phi=Pi/2)
          you need to specify an axis ratio here.

    Return
    ------
    imax3d : float
          Maximum amplitude of the 3D Gaussian
    sig3d : float
          Sigma (dispersion) of the 3D Gaussian
    qxz : float
          Axis ratio (X/Y) of the 3D Gaussian
    qyz : float
          Axis ratio (Y/Y) of the 3D Gaussian
    pa : float
          Position Angle of the Gaussian (starting from North=Vertical up axis,
          going positive counter-clockwise). In degrees.
    euler_angles: array or tuple of 3 floats for triaxial case
          For prolate and oblate cases, one euler_angle (the inclination)
          is needed.
          Viewing Angles (Phi, Theta, Psi). All in degrees. 
          Default is [0.,90.,0.] (edge-on).
    axi : boolean
          Axisymmetric or not (qxz = qyz)
    """

    # Defining the error output if something goes wrong
    error_output = 0., 0., 0., 0., 0., 0., False

    # Check which geometry we wish
    geometries = kwargs.get("euler_angles", 'oblate')
    if geometry not in geometries:
        print("ERROR: you should choose a geometry in the following list: ", 
                geometries)
        return error_output

    # Euler Angles transformed in radian and checked
    if 'inclination' in kwargs :
        inclination = np.degtorad(np.float(kwargs.get('inclination')))
        euler_angles = [0., inclination, 0.]
        if verbose:
            print("Found an inclination. Other Euler angles will be ignored")
    elif 'euler_angles' in kwargs : 
        euler_angles = np.degtorad(np.float(kwargs.get('euler_angles', 
                                   [0., 90., 0.,])))
        if size(euler_angles) != 3 :
            print("ERROR: euler_angles should always" 
                    "have 3 values (tuple or array)")
            return error_output
        inclination = euler_angles[1]
    else :
        print("ERROR: you should set either inclination or euler_angles.")
        return error_output

    # Geometry of the system for deprojection
    geometry = kwargs.get("geometry", 'oblate')

    # Optional parameters in case this is useful
    # By default, set to 1.
    qxz = kwargs.get("qxz", 1.0)
    qyz = kwargs.get("qyz", 1.0)

    if verbose :
        print("Geometry will be ", geometry)

    # OBLATE Geometry --------------------------------------------------------
    if geometry == 'oblate' :
        if euler_angles[2] != 0.  or euler_angles[0] != 0 :
            if verbose :
                print("WARNING: second and third euler angle"
                        " - Phi and Psi - will be ignored")
        #-------------------------------------------
        # FACE-ON case
        if inclination == 0. :
            if q2d != 1 :
                print("ERROR: cannot deproject this model"
                    "as component %d does not have axis ratio of 1!")
            if verbose:
                print("WARNING: this is a face-on case"
                        "An arbitrary value for qxz")
        #-------------------------------------------
        # EDGE-ON case
        elif inclination == np.pi/2. :
            if verbose :
                print "Edge-on deprojection\n"
            qxz = q2d
        #-------------------------------------------
        else :
            cosi2 = cos(inclination) * cos(inclination)
            sini2 = sin(inclination) * sin(inclination)
            if cosi2 > q2d**2 :
                if verbose :
                    maxangle = np.arccos(q2d)
                    print("ERROR: cannot deproject the component %d. Max angle is %f" %(i+1, maxangle*180./np.pi))
                    return error_output
            qxz = np.sqrt((q2d**2 - cosi2) / sini2)

        qyz = qxz
        sig3d = sig2d
        imax3d = imax2d *  q2d / (np.sqrt(2. * np.pi) * qxz * sig2d)
        
    elif prolate :
        pass
    elif triaxial :
        pass
