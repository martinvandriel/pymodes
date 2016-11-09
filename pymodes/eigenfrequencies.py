#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Compute eigenfrequencies as the zeroes of the characteristic function for
homogeneous isotropic elastic full sphere with full gravity, both toroidal and
spheroidal.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import numpy as np
from scipy.optimize import brentq
import warnings

from . import spheroidal
from . import toroidal


def analytical_eigen_frequencies(omega_max, omega_delta, l, rho, vs, vp,
                                 R, tol=1e-8, maxiter=100, mode='T',
                                 gravity=True):
    """
    Find the zeros of the determinant of the Matrix in the 'characteristic' or
    'Rayleigh' function according to: Takeuchi & Saito (1972), Eq (98)-(100)
    for spheroidal and Eq (95) for toroidal in 'Methods in computational
    physics, Volume 11'. Uses sampling first to find zero crossings and then
    Brent's method within the resulting intervals.

    :type omega_max: float
    :param omega_max: maximum frequency to look for zeros
    :type omega_delta: float
    :param omega_delta: spacing for initial sampling in finding zero crossings

    for other paramters see analytical_bc
    """

    # intial sampling of the characteristic function
    omega = np.arange(0., omega_max, omega_delta)

    if mode.upper() == 'T':
        module = toroidal
    elif mode.upper() == 'S':
        module = spheroidal
    else:
        raise ValueError('mode needs to be S or T')

    det = module.analytical_characteristic_function(
        omega=omega, l=l, rho=rho, vs=vs, vp=vp, R=R, gravity=gravity)

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
            f=module.analytical_characteristic_function, a=omega_a[i],
            b=omega_b[i], args=(l, rho, vs, vp, R, gravity), xtol=tol,
            maxiter=maxiter, disp=True)

    return eigen_frequencies


def analytical_eigen_frequencies_catalogue(omega_max, omega_delta, lmax, rho,
                                           vs, vp, R, tol=1e-8, maxiter=100,
                                           mode='T', gravity=True):

    catalogue = []
    for l in np.arange(1, lmax+1):
        catalogue.append(
            analytical_eigen_frequencies(
                omega_max, omega_delta, l, rho, vs, vp, R, tol=tol,
                maxiter=maxiter, mode=mode, gravity=gravity))

    maxn = max([len(x) for x in catalogue])

    catalogue_array = np.empty((lmax, maxn+1)) * np.nan
    for i, overtones in enumerate(catalogue):
        ofs = int(i == 0)
        catalogue_array[i, ofs:ofs+len(overtones)] = overtones

    return catalogue_array
