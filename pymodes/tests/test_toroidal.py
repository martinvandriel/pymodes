#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Tests for the toroidal functions.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import numpy as np

from .. import toroidal


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
