#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Compute eigenfrequencies of toroidal modes in a homogeneous isotropic elastic
full sphere.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import numpy as np
from scipy.special import spherical_jn


def analytical_bc(omega, l, rho, vs, vp, R):
    """
    Compute the the 'characteristic' function according
    to: Takeuchi & Saito (1972), in 'Methods in computational
    physics, Volume 11', Eq (95)

    :type omega: float, scalar or array like
    :param omega: frequency at which to compute the matrix
    :type l: int
    :param l: angular order
    :type rho: float
    :param rho: density in kg / m ** 3
    :type vs: float
    :param float: s-wave velocity in m / s
    :type R: float
    :param R: radius of the sphere
    """

    # Auxiliar variables
    mu = rho * vs ** 2
    omega = np.array(omega)
    k = omega / vs

    j_n = spherical_jn(l, k * R)
    j_np1 = spherical_jn(l+1, k * R)

    y2 = mu / R * ((l - 1) * j_n - k * R * j_np1)

    return y2


def analytical_characteristic_function(omega, l, rho, vs, vp, R):
    """
    Compute the 'characteristic' or function - for toroidal modes this is just
    a wrapper to analytical_bc, see Takeuchi & Saito Eq (77)

    for paramters see analytical_bc
    """

    return analytical_bc(omega=omega, l=l, rho=rho, vs=vs, vp=vp, R=R)
