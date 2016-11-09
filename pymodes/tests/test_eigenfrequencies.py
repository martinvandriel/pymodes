#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Tests for the eigenfrequency rootfinding functions.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import numpy as np

from .. import eigenfrequencies


def test_analytical_eigen_frequencies_branch():

    freq = eigenfrequencies.analytical_eigen_frequencies_branch(
        omega_max=0.01, omega_delta=0.00001, l=10, rho=1e3, vs=1e3, vp=1.7e3,
        R=6371e3, mode='T')

    freq_ref = np.array([0.00185089, 0.00264984, 0.00325404, 0.00381579,
                         0.00435724, 0.00488677, 0.00540855, 0.00592491,
                         0.00643731, 0.00694673, 0.00745382, 0.00795906,
                         0.00846281, 0.00896532, 0.00946681, 0.00996743])

    np.testing.assert_allclose(freq, freq_ref, atol=1e-8)

    freq = eigenfrequencies.analytical_eigen_frequencies_branch(
        omega_max=0.01, omega_delta=0.00001, l=10, rho=1e3, vs=1e3, vp=1.7e3,
        R=6371e3, mode='S')

    freq_ref = np.array([0.00219016, 0.00346125, 0.00459209, 0.00565203,
                         0.006101, 0.00615307, 0.00668435, 0.00699992,
                         0.00718962, 0.00770299, 0.00789299, 0.00820498,
                         0.00871665, 0.00877285, 0.00921247, 0.00966211,
                         0.00970776])

    np.testing.assert_allclose(freq, freq_ref, atol=1e-8)

    # test values from Dahlen & Tromp, section 8.7.4, for l = 1
    freq = eigenfrequencies.analytical_eigen_frequencies_branch(
        omega_max=0.002, omega_delta=0.00001, l=1, rho=1e3, vs=1e3, vp=1.7e3,
        R=6371e3, mode='T')

    np.testing.assert_allclose(freq / 1e3 * 6371e3, [5.76, 9.10, 12.32],
                               atol=1e-2)
