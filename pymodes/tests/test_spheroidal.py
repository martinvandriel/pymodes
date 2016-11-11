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
from .. import eigenfrequencies


def test_analytical_bc():

    m = spheroidal.analytical_bc(omega=0.01, l=10, rho=1e3, vs=1e3, vp=1.7e3,
                                 R=6371e3)

    m_ref = np.array([[-5.98712990e-01, -4.52263515e-04, 4.43462902e-03],
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

    det_ref = np.array([np.nan, -2.87907214e-23, -7.54565054e-17,
                        4.10944167e-16, -1.28911838e-15, 6.69131228e-14,
                        -2.80916424e-13, 2.08765859e-13, 9.94475582e-13,
                        -2.37919436e-12])

    np.testing.assert_allclose(det, det_ref)


def test_integrate_radial():
    # for homogeneous model, check eigenfrequencies

    freq = eigenfrequencies.analytical_eigen_frequencies(
        omega_max=0.01, omega_delta=0.00001, l=10, rho=1e3, vs=1e3, vp=1.7e3,
        R=6371e3, mode='S', gravity=False)

    for f in freq[:5]:
        r, Y2 = spheroidal.integrate_radial(
            rho=1e3, vs=1e3, vp=1.7e3, R=6371e3, l=10, omega=f, r_0=1.,
            nsamp_per_layer=100, rtol=1e-10)

        assert abs(Y2[-1] / Y2.ptp()) < 2e-5
