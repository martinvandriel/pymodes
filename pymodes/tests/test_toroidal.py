#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Tests for the toroidal functions.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import inspect
import numpy as np
import os
import pymesher

from .. import toroidal
from .. import eigenfrequencies


# Most generic way to get the data directory.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


def test_analytical_bc():

    m = toroidal.analytical_bc(omega=0.01, l=10, rho=1e3, vs=1e3, vp=1.7e3,
                               R=6371e3)

    m_ref = 31.6779492015
    np.testing.assert_allclose(m, m_ref)

    # check vectorization on omega
    omegas = [0.01, 0.02, 0.025]
    m = toroidal.analytical_bc(omega=omegas, l=10, rho=1e3, vs=1e3, vp=1.7e3,
                               R=6371e3)

    for i, omega in enumerate(omegas):
        m_ref = toroidal.analytical_bc(
            omega=omega, l=10, rho=1e3, vs=1e3, vp=1.7e3, R=6371e3)
        np.testing.assert_allclose(m[i], m_ref)


def test_dy_dr():

    model = pymesher.model.read(os.path.join(DATA_DIR, 'homo_model.bm'))

    r = 1e6
    y = np.ones(2)
    l = 10
    omega = 0.01

    dy_dr = toroidal.dy_dr(r, y, model, l, omega)
    dy_dr_ref = np.array([1.001e-06, 0.0079969999999999659])

    np.testing.assert_allclose(dy_dr, dy_dr_ref, rtol=1e-15)


def test_dy_dr_homo():

    r = 1e6
    y = np.ones(4)
    rho = 1e3
    vs = 1e3
    l = 10
    omega = 0.01

    L = rho * vs ** 2
    N = rho * vs ** 2

    dy_dr = toroidal.dy_dr_homo(r, y, L, N, rho, l, omega)
    dy_dr_ref = np.array([1.001e-06, 0.0079969999999999])

    np.testing.assert_allclose(dy_dr, dy_dr_ref, rtol=1e-13)


def test_y_initial_conditions():

    r = np.linspace(100., 6e6, 5)
    rho = 1e3
    vs = 1e3
    l = 10
    omega = 0.01

    initial_conditions = toroidal.y_initial_conditions(r, vs, rho, l, omega)

    initial_conditions_ref = np.array(
        [7.27309179e-41, 1.85438550e-03, -1.45440126e-02, -1.75973092e-02,
         1.58227194e-02, 6.54578258e-33, -5.69124694e+02, -2.82438438e+02,
         1.44818262e+02, 5.01738322e+01])

    np.testing.assert_allclose(np.array(initial_conditions).flatten(),
                               initial_conditions_ref, rtol=1e-8)


def test_integrate_radial():
    model = pymesher.model.read(os.path.join(DATA_DIR, 'homo_model.bm'))

    freq = eigenfrequencies.analytical_eigen_frequencies(
        omega_max=0.01, omega_delta=0.00001, l=10, rho=1e3, vs=1e3, vp=1.7e3,
        R=6371e3, mode='T')

    # make sure, the stress is zero at the surface
    # subtracting a tiny bit from the frequency to check mode counting as well
    for i, f in enumerate(freq[:5] - 1e-11):
        r, y1, y2, count = toroidal.integrate_radial(
            model=model, l=10, omega=f, r_0=1e3, nsamp_per_layer=100,
            rtol=1e-10)

        np.testing.assert_allclose(y2[-1] / np.max(y2), 0., atol=1e-4)

        r, y1, y2, count = toroidal.integrate_radial(
            rho=1e3, vs=1e3, R=6371e3, l=10, omega=f, r_0=1e3,
            nsamp_per_layer=100, rtol=1e-10)

        np.testing.assert_allclose(y2[-1] / np.max(y2), 0., atol=1e-4)

        assert count == i
