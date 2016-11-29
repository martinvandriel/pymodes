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

    # l = 1 has a trivial solution
    catalogue_array[0, 0] = 0.

    for i, overtones in enumerate(catalogue):
        ofs = int(i == 0)
        catalogue_array[i, ofs:ofs+len(overtones)] = overtones

    return catalogue_array


def integrate_eigen_frequencies(
   omega_max, l, model=None, rho=None, vs=None, vp=None, R=None, r_0=None,
   omega_min=0., integrator_rtol=1e-10, integrator_nsteps=100000,
   nsamp_per_layer=1, rootfinder_tol=1e-8, rootfinder_maxiter=100, mode='T',
   gravity=False, return_n=False):

    if mode.upper() == 'T':
        integrate_radial = toroidal.integrate_radial
        # compute the modes of the outermost solid shell
        if model is not None and not model.get_fluid_regions() == []:
            r_0 = model.get_fluid_regions()[-1][1] * model.scale + 1e-3
    elif mode.upper() == 'S':
        integrate_radial = spheroidal.integrate_radial_minor
        if gravity:
            raise NotImplementedError('gravity')
        if r_0 is None:
            r_0 = 0.
    else:
        raise ValueError('mode needs to be S or T')

    omega = np.array([omega_min, omega_max])

    # a wrapper to radial integration that returns the mode count
    def mode_count(omega):
        _, _, _, count, _ = integrate_radial(
            omega, l=l, rho=rho, vs=vs, vp=vp, R=R, model=model,
            nsteps=integrator_nsteps, rtol=integrator_rtol, r_0=r_0,
            nsamp_per_layer=nsamp_per_layer, return_compat_mode=True)

        return count

    overtone = np.array([mode_count(omega_min), mode_count(omega_max)])

    # first bracket all eigenfrequencies in the interval [omega_min, omega_max]
    i = 0
    while i < len(omega) - 1:
        # split the interval into half until it contains at max a single
        # eigenfrequency
        while overtone[i+1] > overtone[i] + 1:
            new_omega = (omega[i+1] + omega[i]) / 2
            omega = np.insert(omega, i + 1, new_omega)
            count = mode_count(new_omega)
            overtone = np.insert(overtone, i + 1, count)

        # continue with next interval
        i += 1

    # remove negative overtones resulting from frequencies with turning point
    # outside the planet
    # if np.any(overtone < 0):
    #     raise RuntimeError('should not have happened.')
    mask = overtone >= 0
    overtone = overtone[mask]
    omega = omega[mask]

    # remove intervals not containing an eigenfrequency
    mask = np.diff(overtone) > 0
    omega_a = omega[:-1][mask]
    omega_b = omega[1:][mask]

    nf = len(omega_a)
    eigen_frequencies = np.zeros(nf)

    # a wrapper to radial integration that returns the secular function
    def secular_function(omega):
        _, _, bc, _, _ = integrate_radial(
            omega, l=l, rho=rho, vs=vs, vp=vp, R=R, model=model,
            nsteps=integrator_nsteps, rtol=integrator_rtol, r_0=r_0,
            nsamp_per_layer=nsamp_per_layer, return_compat_mode=True)
        return bc[-1]

    # loop over intervals and converge to tolerance using Brent's method
    for i in np.arange(nf):
        eigen_frequencies[i] = brentq(
            f=secular_function, a=omega_a[i], b=omega_b[i],
            xtol=rootfinder_tol, maxiter=rootfinder_maxiter, disp=True)

    if return_n:
        # TODO: should this not be equal to overone ??
        n = np.zeros(nf, dtype='int')
        for i in np.arange(nf):
            _, _, _, _, n[i] = integrate_radial(
                omega, l, rho, vs, R, model, nsteps=integrator_nsteps,
                rtol=integrator_rtol, r_0=r_0, nsamp_per_layer=nsamp_per_layer,
                return_compat_mode=True)

        return eigen_frequencies, n

    else:
        return eigen_frequencies


def integrate_eigen_frequencies_catalogue(
   omega_max, lmax, model=None, rho=None, vs=None, vp=None, R=None, r_0=None,
   omega_min=0., integrator_rtol=1e-10, integrator_nsteps=100000,
   nsamp_per_layer=10, rootfinder_tol=1e-8, rootfinder_maxiter=100, mode='T',
   gravity=True):

    catalogue = []
    for l in np.arange(1, lmax+1):
        catalogue.append(
            integrate_eigen_frequencies(omega_max, l, model, rho, vs, vp, R,
                                        r_0, omega_min, integrator_rtol,
                                        integrator_nsteps, nsamp_per_layer,
                                        rootfinder_tol, rootfinder_maxiter,
                                        mode, gravity))

    maxn = max([len(x) for x in catalogue])

    catalogue_array = np.empty((lmax, maxn)) * np.nan
    for i, overtones in enumerate(catalogue):
        catalogue_array[i, :len(overtones)] = overtones

    return catalogue_array
