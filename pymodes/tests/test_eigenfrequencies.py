#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Tests for the eigenfrequency rootfinding functions.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import inspect
import numpy as np
import os
import pymesher

from .. import eigenfrequencies


# Most generic way to get the data directory.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


def test_analytical_eigen_frequencies():

    freq = eigenfrequencies.analytical_eigen_frequencies(
        omega_max=0.01, omega_delta=0.00001, l=10, rho=1e3, vs=1e3, vp=1.7e3,
        R=6371e3, mode='T')

    freq_ref = np.array([0.00185089, 0.00264984, 0.00325404, 0.00381579,
                         0.00435724, 0.00488677, 0.00540855, 0.00592491,
                         0.00643731, 0.00694673, 0.00745382, 0.00795906,
                         0.00846281, 0.00896532, 0.00946681, 0.00996743])

    np.testing.assert_allclose(freq, freq_ref, atol=1e-8)

    freq = eigenfrequencies.analytical_eigen_frequencies(
        omega_max=0.01, omega_delta=0.00001, l=10, rho=1e3, vs=1e3, vp=1.7e3,
        R=6371e3, mode='S')

    freq_ref = np.array([0.00261509, 0.00304317, 0.00341088, 0.00389115,
                         0.00417137, 0.00456348, 0.00497069, 0.00519278,
                         0.00562593, 0.00597034, 0.00619746, 0.00665583,
                         0.00692063, 0.0072023, 0.00766575, 0.00784553,
                         0.0082065, 0.00864818, 0.00877123, 0.00920807,
                         0.00957729, 0.00973326])

    np.testing.assert_allclose(freq, freq_ref, atol=1e-8)

    # test values from Dahlen & Tromp, section 8.7.4, for l = 1
    freq = eigenfrequencies.analytical_eigen_frequencies(
        omega_max=0.002, omega_delta=0.00001, l=1, rho=1e3, vs=1e3, vp=1.7e3,
        R=6371e3, mode='T')

    np.testing.assert_allclose(freq / 1e3 * 6371e3, [5.76, 9.10, 12.32],
                               atol=1e-2)


def test_analytical_eigen_frequencies_catalogue():

    cat = eigenfrequencies.analytical_eigen_frequencies_catalogue(
        omega_max=0.01, omega_delta=0.00001, lmax=10, rho=1e3, vs=1e3,
        vp=1.7e3, R=6371e3, mode='T')

    cat_ref = np.array([0., 0.00590478, 0.00165038, 0.00713279, 0.00288998,
                        0.00835924, 0.00411931, 0.00958474, 0.00534555, np.nan,
                        0.00657045, 0.00209772, 0.00779465, 0.00341957,
                        0.00901844, 0.00468094, np.nan, 0.00592491, np.nan])

    np.testing.assert_allclose(cat.flatten()[::11], cat_ref, atol=1e-8)


def test_integrate_eigen_frequencies():

    # compare to analytical solution
    l = 5
    rho = 1e3
    vs = 1e3
    vp = 1.7e3
    R = 6371e3
    omega_max = 0.01

    freq_ref = eigenfrequencies.analytical_eigen_frequencies(
        omega_max=omega_max, omega_delta=0.00001, l=l, rho=rho, vs=vs, vp=vp,
        R=6371e3, mode='T')

    freq = eigenfrequencies.integrate_eigen_frequencies(
        omega_max, l, rho=rho, vs=vs, vp=vp, R=R, mode='T',
        integrator_rtol=1e-8, rootfinder_tol=1e-8)

    np.testing.assert_allclose(freq, freq_ref, atol=1e-8)

    # CI
    model = pymesher.model.built_in('prem_iso')

    freq = eigenfrequencies.integrate_eigen_frequencies(
        omega_max=0.04, l=20, model=model, mode='T', integrator_rtol=1e-7,
        rootfinder_tol=1e-6)

    freq_ref = np.array([0.0175524, 0.02627687, 0.03191921, 0.03658574])
    np.testing.assert_allclose(freq, freq_ref, atol=1e-8)


def test_integrate_eigen_frequencies_catalogue():

    # compare to analytical solution
    lmax = 3
    rho = 1e3
    vs = 1e3
    vp = 1.7e3
    R = 6371e3
    omega_max = 0.005
    omega_delta = 0.00001

    model = pymesher.model.read(os.path.join(DATA_DIR, 'homo_model.bm'))

    ref_cat_t = eigenfrequencies.analytical_eigen_frequencies_catalogue(
        omega_max, omega_delta, lmax, rho, vs, vp, R, mode='T')

    cat_t = eigenfrequencies.integrate_eigen_frequencies_catalogue(
        omega_max, lmax, model=model, integrator_rtol=1e-6,
        rootfinder_tol=1e-6)

    np.testing.assert_allclose(ref_cat_t, cat_t, atol=1e-6)

    # CI
    omega_max = 0.005 * 2 * np.pi
    model = pymesher.model.built_in('prem_iso')

    cat_t = eigenfrequencies.integrate_eigen_frequencies_catalogue(
        omega_max, lmax, model=model, integrator_rtol=1e-6,
        rootfinder_tol=1e-6)

    ref_cat_t = np.array([0., 0.00782317, 0.01386107, 0.020281, 0.02723831,
                          0.00240334, 0.00835497, 0.01413261, 0.02047369,
                          0.02737895, 0.00371385, 0.00910818, 0.01453733,
                          0.02076128, 0.02758907])

    np.testing.assert_allclose(ref_cat_t, cat_t.flatten(), atol=1e-7)
