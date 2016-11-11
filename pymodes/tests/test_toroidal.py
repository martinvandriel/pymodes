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


def test_integrate_radial():
    model = pymesher.model.read(os.path.join(DATA_DIR, 'homo_model.bm'))

    freq = eigenfrequencies.analytical_eigen_frequencies(
        omega_max=0.01, omega_delta=0.00001, l=10, rho=1e3, vs=1e3, vp=1.7e3,
        R=6371e3, mode='T')

    r, y1, y2 = toroidal.integrate_radial(model=model, l=10, omega=freq[0],
                                          r_0=1., nsamp_per_layer=100,
                                          rtol=1e-10)

    np.testing.assert_allclose(y2[-1], 0., atol=1e-5)

    r, y1, y2 = toroidal.integrate_radial(
        rho=1e3, vs=1e3, R=6371e3, l=10, omega=freq[0], r_0=1.,
        nsamp_per_layer=100, rtol=1e-10)

    np.testing.assert_allclose(y2[-1], 0., atol=1e-5)
