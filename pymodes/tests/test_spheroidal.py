#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Tests for the spheroidal functions.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import inspect
import numpy as np
import os
import pymesher

from .. import spheroidal
from .. import eigenfrequencies


# Most generic way to get the data directory.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


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


def test_dy_dr():

    model = pymesher.model.read(os.path.join(DATA_DIR, 'homo_model.bm'))

    r = 1e6
    y = np.ones(4)
    l = 10
    omega = 0.01

    dy_dr = spheroidal.dy_dr(r, y, model, l, omega)
    dy_dr_ref = np.array([3.3259861591695492e-05, -0.44892944636678184,
                          1.0000000000000003e-09, 0.18251572318339085])

    np.testing.assert_allclose(dy_dr, dy_dr_ref, rtol=1e-15)


def test_dy_dr_homo():

    r = 1e6
    y = np.ones(4)
    rho = 1e3
    vs = 1e3
    vp = 1.7e3
    l = 10
    omega = 0.01
    eta = 1.

    A = rho * vp ** 2
    C = rho * vp ** 2
    L = rho * vs ** 2
    N = rho * vs ** 2
    F = eta * (A - 2 * L)

    dy_dr = spheroidal.dy_dr_homo(r, y, A, C, L, N, F, rho, l, omega)
    dy_dr_ref = np.array([3.3259861591695492e-05, -0.44892944636678184,
                          1.0000000000000003e-09, 0.18251572318339085])

    np.testing.assert_allclose(dy_dr, dy_dr_ref, rtol=1e-15)


def test_y_initial_conditions():

    r = np.linspace(100., 6e6, 5)
    rho = 1e3
    vs = 1e3
    vp = 1.7e3
    l = 10
    omega = 0.01

    initial_conditions = spheroidal.y_initial_conditions(
        r, vp, vs, rho, l, omega, solution=1)

    initial_conditions_ref = np.array(
        [3.60769538e-44, 1.33056077e-07, 1.51048460e-07, 2.00560505e-07,
         -1.12997271e-07, 6.49385166e-36, -4.29159162e-04, 4.11606972e-03,
         -1.21953478e-03, -1.94376535e-03, 3.60769539e-45, 2.22093800e-08,
         -1.90471138e-08, 2.59592054e-09, 3.58419562e-09, 6.49385168e-37,
         1.47788206e-04, 1.13395159e-04, 8.79837708e-05, -3.88604888e-05])

    np.testing.assert_allclose(np.array(initial_conditions).flatten(),
                               initial_conditions_ref, rtol=1e-8)

    initial_conditions = spheroidal.y_initial_conditions(
        r, vp, vs, rho, l, omega, solution=2)

    initial_conditions_ref = np.array(
        [-8.00040097e-41, -1.35981471e-07, 5.33271573e-07, 4.30154057e-07,
         -2.90083189e-07, -1.44007217e-32, 8.34674484e-02, 2.07118069e-02,
         -7.07996457e-03, -1.83970718e-03, -8.00040093e-42, 5.66652303e-07,
         2.92134285e-07, -1.36997279e-07, -5.54480720e-08, -1.44007217e-33,
         -7.51359643e-04, -1.29364574e-03, -1.50766506e-03, 1.50406023e-03])

    np.testing.assert_allclose(np.array(initial_conditions).flatten(),
                               initial_conditions_ref, rtol=1e-8)


def test_dY_dr():

    model = pymesher.model.read(os.path.join(DATA_DIR, 'homo_model.bm'))

    r = 1e6
    Y = np.ones(5)
    l = 10
    omega = 0.01

    dY_dr = spheroidal.dY_dr(r, Y, model, l, omega)
    dY_dr_ref = np.array([3.8473702422145353e-07, -0.27282791349480956,
                          -0.35535985467128006, 0.18574663356401369,
                          0.093533947096885811])

    np.testing.assert_allclose(dY_dr, dY_dr_ref, rtol=1e-15)


def test_dY_dr_homo():

    r = 1e6
    Y = np.ones(5)
    rho = 1e3
    vs = 1e3
    vp = 1.7e3
    l = 10
    omega = 0.01
    eta = 1.

    A = rho * vp ** 2
    C = rho * vp ** 2
    L = rho * vs ** 2
    N = rho * vs ** 2
    F = eta * (A - 2 * L)

    dY_dr = spheroidal.dY_dr_homo(r, Y, A, C, L, N, F, rho, l, omega)
    dY_dr_ref = np.array([3.8473702422145332e-07, -0.27282791349480973,
                          -0.35535985467128023, 0.18574663356401383,
                          0.093533947096885825])

    np.testing.assert_allclose(dY_dr, dY_dr_ref, rtol=1e-15)


def test_Y_initial_conditions():

    r = np.linspace(100., 6e6, 5)
    rho = 1e3
    vs = 1e3
    vp = 1.7e3
    l = 10
    omega = 0.01

    initial_conditions = spheroidal.Y_initial_conditions(
        r, vp, vs, rho, l, omega)

    initial_conditions_ref = np.array(
        [-5.45700726e-10, -2.59652088e+00, -5.34431413e-01, -2.56060519e+00,
         -7.02615278e-01, -1.07893649e+07, 3.97774715e+08, 7.55453374e+07,
         2.20443409e+08, 2.88062824e+08, -2.00541598e-01, -3.65803272e+05,
         -9.19048376e+03, -8.01841974e+04, 3.42374398e+04, -7.99950762e-02,
         2.64486131e+03, 2.51911440e+03, -3.04684490e+04, 1.74304972e+04,
         7.81719705e-02, -6.94337667e+04, 1.57221672e+04, -1.66079503e+04,
         1.10003148e+04])

    np.testing.assert_allclose(np.array(initial_conditions).flatten(),
                               initial_conditions_ref, rtol=1e-8)


def test_integrate_radial():
    # for homogeneous model, check eigenfunctions

    l = 10
    rho = 1e3
    vs = 1e3
    vp = 1.7e3
    R = 6371e3

    rtol = 1e-5

    freq = eigenfrequencies.analytical_eigen_frequencies(
        omega_max=0.01, omega_delta=0.00001, l=l, rho=rho, vs=vs, vp=vp,
        R=R, mode='S', gravity=False)

    r_0 = 1e6
    model = pymesher.model.read(os.path.join(DATA_DIR, 'homo_model.bm'))

    for f in freq[:5]:
        r, y11, y21, y31, y41 = spheroidal.integrate_radial(
            rho=rho, vs=vs, vp=vp, R=R, l=l, omega=f, r_0=r_0,
            nsamp_per_layer=1000, rtol=rtol, nsteps=10000, solution=1)

        y11_ref, y21_ref, y31_ref, y41_ref = spheroidal.y_initial_conditions(
            r, vp, vs, rho, l, f, 1)

        np.testing.assert_allclose(y11 / y11[-1], y11_ref / y11[-1], atol=rtol)
        np.testing.assert_allclose(y21 / y21[-1], y21_ref / y21[-1], atol=rtol)
        np.testing.assert_allclose(y31 / y31[-1], y31_ref / y31[-1], atol=rtol)
        np.testing.assert_allclose(y41 / y41[-1], y41_ref / y41[-1], atol=rtol)

        r, y11, y21, y31, y41 = spheroidal.integrate_radial(
            model=model, l=l, omega=f, r_0=r_0,
            nsamp_per_layer=1000, rtol=rtol, nsteps=10000, solution=1)

        np.testing.assert_allclose(y11 / y11[-1], y11_ref / y11[-1], atol=rtol)
        np.testing.assert_allclose(y21 / y21[-1], y21_ref / y21[-1], atol=rtol)
        np.testing.assert_allclose(y31 / y31[-1], y31_ref / y31[-1], atol=rtol)
        np.testing.assert_allclose(y41 / y41[-1], y41_ref / y41[-1], atol=rtol)

        r, y12, y22, y32, y42 = spheroidal.integrate_radial(
            rho=rho, vs=vs, vp=vp, R=R, l=l, omega=f, r_0=r_0,
            nsamp_per_layer=1000, rtol=rtol, nsteps=10000, solution=2)

        y12_ref, y22_ref, y32_ref, y42_ref = spheroidal.y_initial_conditions(
            r, vp, vs, rho, l, f, 2)

        np.testing.assert_allclose(y12 / y12[-1], y12_ref / y12[-1], atol=rtol)
        np.testing.assert_allclose(y22 / y22[-1], y22_ref / y22[-1], atol=rtol)
        np.testing.assert_allclose(y32 / y32[-1], y32_ref / y32[-1], atol=rtol)
        np.testing.assert_allclose(y42 / y42[-1], y42_ref / y42[-1], atol=rtol)

        r, y12, y22, y32, y42 = spheroidal.integrate_radial(
            model=model, l=l, omega=f, r_0=r_0,
            nsamp_per_layer=1000, rtol=rtol, nsteps=10000, solution=2)

        np.testing.assert_allclose(y12 / y12[-1], y12_ref / y12[-1], atol=rtol)
        np.testing.assert_allclose(y22 / y22[-1], y22_ref / y22[-1], atol=rtol)
        np.testing.assert_allclose(y32 / y32[-1], y32_ref / y32[-1], atol=rtol)
        np.testing.assert_allclose(y42 / y42[-1], y42_ref / y42[-1], atol=rtol)


def test_integrate_radial_minor():
    # for homogeneous model, check eigenfrequencies

    freq = eigenfrequencies.analytical_eigen_frequencies(
        omega_max=0.01, omega_delta=0.00001, l=10, rho=1e3, vs=1e3, vp=1.7e3,
        R=6371e3, mode='S', gravity=False)

    for f in freq[:5]:
        r, Y2 = spheroidal.integrate_radial_minor(
            rho=1e3, vs=1e3, vp=1.7e3, R=6371e3, l=10, omega=f, r_0=1.,
            nsamp_per_layer=10, rtol=1e-5)

        assert abs(Y2[-1] / Y2.ptp()) < 2e-5

    model = pymesher.model.read(os.path.join(DATA_DIR, 'homo_model.bm'))

    for f in freq[:5]:
        r, Y2 = spheroidal.integrate_radial_minor(
            model=model, l=10, omega=f, r_0=1., nsamp_per_layer=10,
            rtol=1e-5)

        assert abs(Y2[-1] / Y2.ptp()) < 2e-5
