#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Tests for the spheroidal functions.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import numpy as np

from .. import spheroidal


def test_analytical_bc():

    m = spheroidal.analytical_bc(omega=0.01, l=10, rho=1e3, vs=1e3, vp=1.7e3,
                                 R=6371e3)

    m_ref = np.array([[6.62305365e-01, -1.92070366e-03, 2.82530215e+04],
                      [1.36912996e-04, -1.45827457e-03, 4.43462902e-04],
                      [8.65248156e-12, -2.28496427e-13, -3.21720079e-10]])
    np.testing.assert_allclose(m, m_ref)

    # check vectorization on omega
    omegas = [0.01, 0.02, 0.025]
    m = spheroidal.analytical_bc(omega=omegas, l=10, rho=1e3, vs=1e3, vp=1.7e3,
                                 R=6371e3)

    for i, omega in enumerate(omegas):
        m_ref = spheroidal.analytical_bc(
            omega=omega, l=10, rho=1e3, vs=1e3, vp=1.7e3, R=6371e3)
        np.testing.assert_allclose(m[:, :, i], m_ref)


def test_analytical_characteristic_function():

    omega = np.linspace(0.0, .015, 10)
    det = spheroidal.analytical_characteristic_function(
        omega=omega, l=10, rho=1e3, vs=1e3, vp=1.7e3, R=6371e3)

    det_ref = np.array([np.NaN, -9.90890439e-16, 1.33665845e-11,
                        3.00013818e-11, -1.60244314e-11, -1.88052027e-10,
                        3.55914802e-10, -1.26477769e-10, -3.65516534e-10,
                        5.06938358e-10])

    np.testing.assert_allclose(det, det_ref)
