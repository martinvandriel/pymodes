#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Compute eigenfrequencies as the zeroes of the Rayleigh function for a
homogeneous isotropic elastic full sphere with full gravity.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
    Federico Munch
:license:
    None
'''
import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq
import warnings


G = 6.67408e-11


def analytical_bc(omega, l, rho, vs, vp, R, gravity=True):
    """
    Compute the Matrix in the 'characteristic' or 'Rayleigh' function according
    to: Takeuchi & Saito (1972), in 'Methods in computational
    physics, Volume 11'. Eq (98)-(100) for including gravity, Eq (104) - (105)
    for neglecting gravity.

    :type omega: float, scalar or array like
    :param omega: frequency at which to compute the matrix
    :type l: int
    :param l: angular order
    :type rho: float
    :param rho: density in kg / m ** 3
    :type vs: float
    :param float: s-wave velocity in m / s
    :type vp: float
    :param float: p-wave velocity in m / s
    :type R: float
    :param R: radius of the sphere
    :type gravity: Bool
    :param gravity: enable gravity in the calculation
    """

    # Auxiliar variables
    mu = rho * vs ** 2
    # lam = rho * vp ** 2 - 2 * mu
    omega = np.array(omega)
    w2 = omega ** 2

    if gravity:
        gamma = 4. * np.pi * G * rho / 3.

        a = (w2 + 4. * gamma) / vp ** 2 + w2 / vs ** 2
        b = (w2 / vs ** 2) - (w2 + 4. * gamma) / vp ** 2
        c = 4. * l * (l + 1) * gamma ** 2 / (vp ** 2 * vs ** 2)
        d = (b ** 2 + c) ** 0.5

        # - root of k ** 2 - Eq (99) Takeuchi & Saito (1972)
        # for long periods, d > a, which raises a NaN warning that can safely
        # be ignored
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k1 = np.sqrt((a - d) / 2)
        f1 = vs ** 2 / gamma * (k1 ** 2 - w2 / vs ** 2)
        h1 = f1 - (l + 1)
        x1 = k1 * R

        # + root of k ** 2  - Eq (99) Takeuchi & Saito (1972)
        k2 = np.sqrt((a + d) / 2)
        f2 = vs ** 2 / gamma * (k2 ** 2 - w2 / vs ** 2)
        h2 = f2 - (l + 1)
        x2 = k2 * R

        # precompute bessel functions
        j_n_x1 = spherical_jn(l, x1)
        j_np1_x1 = spherical_jn(l+1, x1)
        j_n_x2 = spherical_jn(l, x2)
        j_np1_x2 = spherical_jn(l+1, x2)

        # Compute matrix's elements  - Eq (98 & 100) Takeuchi & Saito (1972)
        # Solutions y2i (named as R by Dahlen & Tromp book)
        y21 = ((vp ** 2 * rho) * f1 * x1 ** 2 * j_n_x1 +
               2 * mu * l * (l-1) * h1 * j_n_x1 +
               2 * mu * (2 * f1 + l * (l + 1)) * x1 * j_np1_x1) / R ** 2

        y22 = ((vp ** 2 * rho) * f2 * x2 ** 2 * j_n_x2 +
               2 * mu * l * (l - 1) * h2 * j_n_x2 +
               2 * mu * (2 * f2 + l * (l + 1)) * x2 * j_np1_x2) / R ** 2

        y23 = 2. * mu * l * (l - 1) / R

        # Solutions y4i (named as S by Dahlen & Tromp book)
        y41 = (mu * x1 ** 2 * j_n_x1 + mu * 2 * (l - 1) * h1 * j_n_x1 -
               mu * 2 * (f1 + 1) * x1 * j_np1_x1) / R ** 2

        y42 = (mu * x2 ** 2 * j_n_x2 + mu * 2 * (l - 1) * h2 * j_n_x2 -
               mu * 2 * (f2 + 1) * x2 * j_np1_x2) / R ** 2

        y43 = 2 * mu * (l - 1) / R ** 2

        # Solutions y5i (named as P by Dahlen & Tromp book) - Necessary to
        # compute y6i (B)
        y51 = 3 * gamma * f1 * j_n_x1
        y52 = 3 * gamma * f2 * j_n_x2
        y53 = (l * gamma - w2)

        # Solutions y6i (named as B by Dahlen & Tromp book)
        y61 = ((2 * l + 1) * y51 - 3 * l * gamma * h1 * j_n_x1) / R
        y62 = ((2 * l + 1) * y52 - 3. * l * gamma * h2 * j_n_x2) / R
        y63 = (2 * l + 1) * y53 / R - 3. * l * gamma / R

        # make sure also frequency independent terms are vectorized in the same
        # way as omega
        y23 *= np.ones_like(omega)
        y43 *= np.ones_like(omega)

        # Build matrix
        return np.array([[y21, y22, y23], [y41, y42, y43], [y61, y62, y63]])

    else:
        raise NotImplementedError()
        return np.array([[y21, y22], [y41, y42]])


def analytical_characteristic_function(omega, l, rho, vs, vp, R):
    """
    Compute the determinant of the Matrix in the 'characteristic' or 'Rayleigh'
    function according to: Takeuchi & Saito (1972) - Eq (98)-(100), in 'Methods
    in computational physics, Volume 11'

    for paramters see analytical_bc
    """

    m = analytical_bc(omega=omega, l=l, rho=rho, vs=vs, vp=vp, R=R)

    # np.linalg.det expects the last two axis to be the matrizes
    # analytical_bc returns the frequency on the last axis, so need to roll it
    # in case
    if m.ndim == 3:
        m = np.rollaxis(m, 2)

    # catch NaN warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.linalg.det(m)


def analytical_eigen_frequencies_branch(omega_max, omega_delta, l, rho, vs, vp,
                                        R, tol=1e-8, maxiter=100):
    """
    Find the zeros of the determinant of the Matrix in the 'characteristic' or
    'Rayleigh' function according to: Takeuchi & Saito (1972) - Eq (98)-(100),
    in 'Methods in computational physics, Volume 11'. Uses sampling first to
    find zero crossings and then Brent's method within the resulting intervals.

    :type omega_max: float
    :param omega_max: maximum frequency to look for zeros
    :type omega_delta: float
    :param omega_delta: spacing for initial sampling in finding zero crossings

    for other paramters see analytical_bc
    """

    # intial sampling of the characteristic function
    omega = np.arange(0., omega_max, omega_delta)
    det = analytical_characteristic_function(
        omega=omega, l=l, rho=rho, vs=vs, vp=vp, R=R)

    # find intervals with zero crossing
    # catch NaN warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idx_zero_crossing = det[1:] * det[:-1] < 0
        # TODO: add a check that we are not in numerical noise

    nf = idx_zero_crossing.sum()

    omega_a = omega[:-1][idx_zero_crossing]
    omega_b = omega[1:][idx_zero_crossing]

    eigen_frequencies = np.zeros(nf)

    # loop over intervals and converge to tolerance using Brent's method
    for i in np.arange(nf):
        eigen_frequencies[i] = brentq(
            f=analytical_characteristic_function, a=omega_a[i], b=omega_b[i],
            args=(l, rho, vs, vp, R), xtol=tol, maxiter=maxiter, disp=True)

    return eigen_frequencies


if __name__ == "__main__":

    omega_max = 0.01
    omega_delta = 0.000001
    omega = np.arange(0., omega_max, omega_delta)

    l = 10
    rho = 1e3
    vs = 1e3
    vp = 1.7e3
    R = 6371e3

    det = analytical_characteristic_function(omega, l, rho, vs, vp, R)

    freq = analytical_eigen_frequencies_branch(
        omega_max, omega_delta, l, rho, vs, vp, R)

    print freq

    import matplotlib.pyplot as plt
    plt.plot(omega, det)
    plt.axhline(0., color='k')

    for f in freq:
        plt.axvline(f, color='k')

    plt.show()