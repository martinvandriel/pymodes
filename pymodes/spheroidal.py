#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Compute eigenfrequencies as the zeroes of the Rayleigh function for a
homogeneous isotropic elastic full sphere with full gravity.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
    Federico Munch
:license:
    None
'''
import numpy as np
from scipy.integrate import ode
from scipy.special import spherical_jn
import warnings

from .start_level import start_level
from .misc import get_radial_sampling, InitError
from .mode_counter import mode_counter

G = 6.67408e-11


def analytical_bc(omega, l, rho, vs, vp, R, gravity=True):
    """
    Compute the Matrix in the 'characteristic' or 'Rayleigh' function according
    to: Takeuchi & Saito (1972), in 'Methods in computational
    physics, Volume 11'. Eq (98)-(100) for including gravity, Eq (104) - (105)
    for neglecting gravity.

    :type omega: float, scalar or array like
    :param omega: frequency at which to compute the matrix
    :type l: int
    :param l: angular order
    :type rho: float
    :param rho: density in kg / m ** 3
    :type vs: float
    :param float: s-wave velocity in m / s
    :type vp: float
    :param float: p-wave velocity in m / s
    :type R: float
    :param R: radius of the sphere
    :type gravity: Bool
    :param gravity: enable gravity in the calculation
    """

    # Auxiliar variables
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2 * mu
    omega = np.array(omega)
    w2 = omega ** 2

    if gravity:
        gamma = 4. * np.pi * G * rho / 3.

        a = (w2 + 4. * gamma) / vp ** 2 + w2 / vs ** 2
        b = (w2 / vs ** 2) - (w2 + 4. * gamma) / vp ** 2
        c = 4. * l * (l + 1) * gamma ** 2 / (vp ** 2 * vs ** 2)
        d = (b ** 2 + c) ** 0.5

        # - root of k ** 2 - Eq (99) Takeuchi & Saito (1972)
        # for long periods, d > a, which raises a NaN warning that can safely
        # be ignored
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k1 = np.sqrt((a - d) / 2)
        f1 = vs ** 2 / gamma * (k1 ** 2 - w2 / vs ** 2)
        h1 = f1 - (l + 1)
        x1 = k1 * R

        # + root of k ** 2  - Eq (99) Takeuchi & Saito (1972)
        k2 = np.sqrt((a + d) / 2)
        f2 = vs ** 2 / gamma * (k2 ** 2 - w2 / vs ** 2)
        h2 = f2 - (l + 1)
        x2 = k2 * R

        # precompute bessel functions
        j_n_x1 = spherical_jn(l, x1)
        j_np1_x1 = spherical_jn(l+1, x1)
        j_n_x2 = spherical_jn(l, x2)
        j_np1_x2 = spherical_jn(l+1, x2)

        # Compute matrix's elements  - Eq (98 & 100) Takeuchi & Saito (1972)
        # Solutions y2i (named as R by Dahlen & Tromp book)
        y21 = (-vp ** 2 * rho * f1 * x1 ** 2 * j_n_x1 +
               2 * mu * l * (l - 1) * h1 * j_n_x1 +
               2 * mu * (2 * f1 + l * (l + 1)) * x1 * j_np1_x1) / R ** 2

        y22 = (-vp ** 2 * rho * f2 * x2 ** 2 * j_n_x2 +
               2 * mu * l * (l - 1) * h2 * j_n_x2 +
               2 * mu * (2 * f2 + l * (l + 1)) * x2 * j_np1_x2) / R ** 2

        y23 = 2 * mu * l * (l - 1) / R ** 2

        # Solutions y4i (named as S by Dahlen & Tromp book)
        y41 = (mu * x1 ** 2 * j_n_x1 + mu * 2 * (l - 1) * h1 * j_n_x1 -
               mu * 2 * (f1 + 1) * x1 * j_np1_x1) / R ** 2

        y42 = (mu * x2 ** 2 * j_n_x2 + mu * 2 * (l - 1) * h2 * j_n_x2 -
               mu * 2 * (f2 + 1) * x2 * j_np1_x2) / R ** 2

        y43 = 2 * mu * (l - 1) / R ** 2

        # Solutions y5i (named as P by Dahlen & Tromp book) - Necessary to
        # compute y6i (B)
        y51 = 3 * gamma * f1 * j_n_x1
        y52 = 3 * gamma * f2 * j_n_x2
        y53 = (l * gamma - w2)

        # Solutions y6i (named as B by Dahlen & Tromp book)
        y61 = ((2 * l + 1) * y51 - 3 * l * gamma * h1 * j_n_x1) / R
        y62 = ((2 * l + 1) * y52 - 3 * l * gamma * h2 * j_n_x2) / R
        y63 = ((2 * l + 1) * y53 - 3 * l * gamma) / R

        # make sure also frequency independent terms are vectorized in the same
        # way as omega
        y23 *= np.ones_like(omega)
        y43 *= np.ones_like(omega)

        # Build matrix
        return np.array([[y21, y22, y23], [y41, y42, y43], [y61, y62, y63]])

    else:
        x1 = omega / vp * R

        j_n_x1 = spherical_jn(l, x1)
        j_np1_x1 = spherical_jn(l+1, x1)

        y21 = (-(lam + 2 * mu) * x1 ** 2 * j_n_x1 +
               2 * mu * (l * (l - 1) * j_n_x1 + 2 * x1 * j_np1_x1)) / R ** 2

        y41 = 2 * mu * ((l - 1) * j_n_x1 - x1 * j_np1_x1) / R ** 2

        x2 = omega / vs * R

        j_n_x2 = spherical_jn(l, x2)
        j_np1_x2 = spherical_jn(l+1, x2)

        y22 = 2 * mu * (
            -l * (l ** 2 - 1) * j_n_x2 + l * (l + 1) * x2 * j_np1_x2) / R ** 2

        y42 = mu * (x2 ** 2 * j_n_x2 - 2 * (l ** 2 - 1) * j_n_x2 -
                    2 * x2 * j_np1_x2) / R ** 2

        return np.array([[y21, y22], [y41, y42]])


def analytical_characteristic_function(omega, l, rho, vs, vp, R, gravity=True):
    """
    Compute the determinant of the Matrix in the 'characteristic' or 'Rayleigh'
    function according to: Takeuchi & Saito (1972) - Eq (98)-(100), in 'Methods
    in computational physics, Volume 11'

    for paramters see analytical_bc
    """

    m = analytical_bc(omega=omega, l=l, rho=rho, vs=vs, vp=vp, R=R,
                      gravity=gravity)

    # np.linalg.det expects the last two axis to be the matrizes
    # analytical_bc returns the frequency on the last axis, so need to roll it
    # in case
    if m.ndim == 3:
        m = np.rollaxis(m, 2)

    # catch NaN warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.linalg.det(m)


def dy_dr(r, y, model, l, omega):
    """
    Takeuchi & Saito (1972), Eq. (91) for general models.
    """

    if model.anisotropic:
        vsv = model.get_native_parameter('VSV', r/model.scale)
        vsh = model.get_native_parameter('VSH', r/model.scale)
        vpv = model.get_native_parameter('VPV', r/model.scale)
        vph = model.get_native_parameter('VPH', r/model.scale)
        eta = model.get_native_parameter('ETA', r/model.scale)
    else:
        vsv = model.get_native_parameter('VS', r/model.scale)
        vpv = model.get_native_parameter('VP', r/model.scale)
        vsh = vsv
        vph = vpv
        eta = 1

    rho = model.get_native_parameter('RHO', r/model.scale)

    # Nolet (2008), p292
    A = rho * vph ** 2
    C = rho * vpv ** 2
    L = rho * vsv ** 2
    N = rho * vsh ** 2
    F = eta * (A - 2 * L)

    return dy_dr_homo(r, y, A, C, L, N, F, rho, l, omega)


def dy_dr_homo(r, y, A, C, L, N, F, rho, l, omega):
    """
    Takeuchi & Saito (1972), Eq. (91) for homogeneous models.
    """

    y1, y2, y3, y4 = y

    a = A - F ** 2 / C - N
    b = A - F ** 2 / C

    c = l * (l + 1) / r ** 2
    d = l * (l + 1) / r

    e = omega ** 2 * rho

    dy1_dr = (-2 * F / (C * r) * y1 +
              1 / C * y2 +
              d * F / C * y3)

    dy2_dr = ((-e + 4 / r ** 2 * a) * y1 +
              2 / r * (F / C - 1) * y2 -
              2 * c * a * y3 +
              d * y4)

    dy3_dr = (-1 / r * y1 +
              1 / r * y3 +
              1 / L * y4)

    dy4_dr = (-2 / r ** 2 * a * y1 -
              F / (C * r) * y2 +
              (-e + c * b - 2 * N / r ** 2) * y3 -
              3 / r * y4)

    return [dy1_dr, dy2_dr, dy3_dr, dy4_dr]


def y_initial_conditions(r1, vp, vs, rho, l, omega, solution=1):
    """
    Takeuchi & Saito (1972), Eq. (104 - 105).
    """
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2 * mu

    if solution == 1:
        x1 = omega / vp * r1

        j_n_x1 = spherical_jn(l, x1)
        j_np1_x1 = spherical_jn(l+1, x1)

        y11 = (l * j_n_x1 - x1 * j_np1_x1) / r1

        y21 = (-(lam + 2 * mu) * x1 ** 2 * j_n_x1 +
               2 * mu * (l * (l - 1) * j_n_x1 + 2 * x1 * j_np1_x1)) / r1 ** 2

        y31 = j_n_x1 / r1

        y41 = 2 * mu * ((l - 1) * j_n_x1 - x1 * j_np1_x1) / r1 ** 2

        return [y11, y21, y31, y41]

    elif solution == 2:
        x2 = omega / vs * r1

        j_n_x2 = spherical_jn(l, x2)
        j_np1_x2 = spherical_jn(l+1, x2)

        y12 = -l * (l + 1) * j_n_x2 / r1

        y22 = 2 * mu * (
            -l * (l ** 2 - 1) * j_n_x2 + l * (l + 1) * x2 * j_np1_x2) / r1 ** 2

        y32 = (-(l + 1) * j_n_x2 + x2 * j_np1_x2) / r1

        y42 = mu * (x2 ** 2 * j_n_x2 - 2 * (l ** 2 - 1) * j_n_x2 -
                    2 * x2 * j_np1_x2) / r1 ** 2

        return [y12, y22, y32, y42]
    else:
        raise ValueError()


def dY_dr(r, Y, model, l, omega):
    """
    Takeuchi & Saito (1972), Eq. (164) for general models.
    """

    if model.anisotropic:
        vsv = model.get_native_parameter('VSV', r/model.scale)
        vsh = model.get_native_parameter('VSH', r/model.scale)
        vpv = model.get_native_parameter('VPV', r/model.scale)
        vph = model.get_native_parameter('VPH', r/model.scale)
        eta = model.get_native_parameter('ETA', r/model.scale)
    else:
        vsv = model.get_native_parameter('VS', r/model.scale)
        vpv = model.get_native_parameter('VP', r/model.scale)
        vsh = vsv
        vph = vpv
        eta = 1

    rho = model.get_native_parameter('RHO', r/model.scale)

    # Nolet (2008), p292
    A = rho * vph ** 2
    C = rho * vpv ** 2
    L = rho * vsv ** 2
    N = rho * vsh ** 2
    F = eta * (A - 2 * L)

    return dY_dr_homo(r, Y, A, C, L, N, F, rho, l, omega)


def dY_dr_homo(r, Y, A, C, L, N, F, rho, l, omega):
    """
    Takeuchi & Saito (1972), Eq. (164) for homogeneous models.
    """

    Y1, Y2, Y3, Y4, Y5 = Y

    a = A - F ** 2 / C - N
    b = A - F ** 2 / C

    c = l * (l + 1) / r ** 2
    d = l * (l + 1) / r

    e = omega ** 2 * rho

    dY1_dr = (1 / r * (1 - 2 * F / C) * Y1 +
              1 / L * Y4 -
              1 / C * Y5)

    dY2_dr = (1 / r * (2 * F / C - 5) * Y2 +
              4 / r ** 2 * a * Y3 +
              (-e + 4 / r ** 2 * a) * Y4 +
              (e - c * b + 2 * N / r ** 2) * Y5)

    dY3_dr = (-2 * c * a * Y1 -
              2 / r * Y3 +
              d * Y4 +
              d * F / C * Y5)

    dY4_dr = ((-e + c * b - 2 * N / r ** 2) * Y1 +
              1 / C * Y2 -
              2 * F / (C * r) * Y3 -
              1 / r * (3 + 2 * F / C) * Y4)

    dY5_dr = ((e - 4 / r ** 2 * a) * Y1 -
              1 / L * Y2 -
              2 / r * Y3 -
              1 / r * (1 - 2 * F / C) * Y5)

    return [dY1_dr, dY2_dr, dY3_dr, dY4_dr, dY5_dr]


def Y_initial_conditions_new(r1, vp, vs, rho, l, omega):
    """
    compute the minor initial conditions from the standard ones via Takeuchi &
    Saito equation (153). Avoids the divion by zero at roots of the Bessel
    functions in the z_l function.
    """
    y11, y21, y31, y41 = y_initial_conditions(r1, vp, vs, rho, l, omega,
                                              solution=1)
    y12, y22, y32, y42 = y_initial_conditions(r1, vp, vs, rho, l, omega,
                                              solution=2)

    Y1 = y11 * y32 - y12 * y31
    Y2 = y21 * y42 - y22 * y41
    Y3 = y11 * y22 - y12 * y21
    Y4 = y11 * y42 - y12 * y41
    Y5 = y31 * y22 - y32 * y21

    return Y1, Y2, Y3, Y4, Y5


def z_l(l, x):
    """
    Takeuchi & Saito (1972), Eq. (96).
    """
    return x * spherical_jn(l+1, x) / spherical_jn(l, x)


def z_l_rec(l, x, nrec=100):
    """
    Takeuchi & Saito (1972), Eq. (9y).
    """
    z = x ** 2 / (2 * (l + nrec) + 3)
    for i in np.arange(nrec):
        li = l + nrec - i
        z = x ** 2 / ((2 * li + 1) - z)
    return z


def Y_initial_conditions(r1, vp, vs, rho, l, omega):
    """
    Takeuchi & Saito (1972), Eq. (165). Can only be used below the first zero
    crossing, otherwise will lead to division by zero errors.
    """

    xa = omega / vp * r1
    xb = omega / vs * r1

    za = 1. / l * z_l_rec(l, xa)
    zb = 1. / (l + 1) * z_l_rec(l, xb)

    mu = rho * vs ** 2

    Y1 = 1. / l * (-za + zb * (za - 1))

    Y2 = mu ** 2 / r1 ** 2 * (
        -4 * (l - 1) * (l + 2) * Y1 +
        xb ** 2 / l * (xb ** 2 / (l * (l + 1)) -
                       2. * (l - 1) * (2 * l + 1) / (l * (l + 1.)) -
                       4. / (l + 1) * za - 2. / l * zb))

    Y3 = mu / r1 * (xb ** 2 / l + 2 * l * (l + 1) * Y1)

    Y4 = mu / r1 * (-2 * Y1 + xb ** 2 / (l * (l + 1)) * (za - 1))

    Y5 = mu / r1 * (xb ** 2 / l ** 2 * (1 - zb) + 4 * Y1)

    return [Y1, Y2, Y3, Y4, Y5]


def integrate_radial(omega, l, rho=None, vs=None, vp=None, R=None, model=None,
                     nsteps=10000, rtol=1e-15, r_0=1e-10, nsamp_per_layer=100,
                     solution=1):
    """
    integrate the spheroidal radial equation Takeuchi & Saito (1972), Eq (91)
    radially, initial conditions assume a homogeneous sphere within the radius
    r_0. Fully solid planets only.
    """

    if model is not None:
        if np.any(model.get_fluid_regions()):
            raise ValueError('Not a fully solid planet!')

        r_in_m = get_radial_sampling(model, nsamp_per_layer, r_0)

        # evaluate model to compute initial values assuming isotropic,
        # homogeneous sphere in the center
        _r = r_in_m[:1] / model.scale
        vp = model.get_elastic_parameter('VP', _r)[0]
        vs = model.get_elastic_parameter('VS', _r)[0]
        rho = model.get_elastic_parameter('RHO', _r)[0]

        integrator = ode(dy_dr)
        integrator.set_f_params(model, l, omega)

    elif rho is not None and vs is not None and vp is not None \
            and R is not None:

        if vs == 0:
            raise ValueError('Not a fully solid planet!')

        r_in_m = np.linspace(r_0, R, nsamp_per_layer+1)

        mu = rho * vs ** 2
        lam = rho * vp ** 2 - 2 * mu

        A = rho * vp ** 2
        C = A
        L = mu
        N = mu
        F = lam

        integrator = ode(dy_dr_homo)
        integrator.set_f_params(A, C, L, N, F, rho, l, omega)
    else:
        raise ValueError('either provide a pymesher model or vs, rho and R')

    # assume to start at a stress free boundary and set initial conditions
    nr = len(r_in_m)
    y1 = np.zeros(nr)
    y2 = np.zeros(nr)
    y3 = np.zeros(nr)
    y4 = np.zeros(nr)

    initial_conditions = y_initial_conditions(r_0, vp, vs, rho, l, omega,
                                              solution=solution)
    y1[0], y2[0], y3[0], y4[0] = initial_conditions

    integrator.set_integrator('dopri5', nsteps=nsteps, rtol=rtol)
    integrator.set_initial_value(initial_conditions, r_0)

    # Do the actual integration
    for i in np.arange(nr - 1):

        integrator.integrate(r_in_m[i+1])

        if not integrator.successful():
            raise RuntimeError(
                "Integration Error while integrating radial equation")

        y1[i+1], y2[i+1], y3[i+1], y4[i+1] = integrator.y

        # avoid float overflows by rescaling if the solution grows big
        rfac = 10. ** int(np.log10(np.max(np.abs([
            y1[i+1], y2[i+1], y2[i+1], y3[i+1]]))))
        if rfac > 1.:
            integrator.set_initial_value(integrator.y / rfac, integrator.t)
            y1 /= rfac
            y2 /= rfac
            y3 /= rfac
            y4 /= rfac

    return r_in_m, y1, y2, y3, y4


def integrate_radial_minor(omega, l, rho=None, vs=None, vp=None, R=None,
                           model=None, nsteps=10000, rtol=1e-15, r_0=0.,
                           nsamp_per_layer=100, return_compat_mode=False):
    """
    integrate the minor vector equation Takeuchi & Saito (1972), Eq (164)
    radially, initial conditions assume a homogeneous sphere within the radius
    r_0. Fully solid planets only.
    """
    if model is not None:
        if np.any(model.get_fluid_regions()):
            raise ValueError('Not a fully solid planet!')

        r_in_m, discontinuities = get_radial_sampling(
            model, nsamp_per_layer, r_0, return_discontinuities=True)

        # find starting radius
        def v_func(_r):
            v = model.get_elastic_parameter('VS', _r / model.scale)
            vp = model.get_elastic_parameter('VP', _r / model.scale)
            mask_fluid = v == 0.
            v[mask_fluid] = vp[mask_fluid]
            return v

        r_start = start_level(v_func, omega, l, r_0, model.scale)

        # make sure to stay below the first discontinuity
        r_start = min(r_start, discontinuities[1] * model.scale)

        # snap to discontinuities, somewhat empirical try to make sure to not
        # miss any zeros in mode counting
        # mask = np.logical_and(discontinuities > 0.,
        #                       discontinuities < r_start)
        # crit = (r_start / (discontinuities[mask] * model.scale) <
        #         1. + 100 * r_start / model.scale / l)
        # if np.any(crit):
        #     r_start = min(r_start, discontinuities[mask][np.argmax(crit)] *
        #                   model.scale)
        #     #print 'snapping r_start to discontinuity, ', r_start

        # evaluate model to compute initial values assuming isotropic,
        # homogeneous sphere in the center
        _r = np.array([r_start]) / model.scale
        vp = model.get_elastic_parameter('VP', _r)[0]
        vs = model.get_elastic_parameter('VS', _r)[0]
        rho = model.get_elastic_parameter('RHO', _r)[0]
        R = model.scale

        integrator = ode(dY_dr)
        integrator.set_f_params(model, l, omega)

    elif rho is not None and vs is not None and vp is not None \
            and R is not None:

        if vs == 0:
            raise ValueError('Not a fully solid planet!')

        r_in_m = np.linspace(r_0, R, nsamp_per_layer+1)

        mu = rho * vs ** 2
        lam = rho * vp ** 2 - 2 * mu

        A = rho * vp ** 2
        C = A
        L = mu
        N = mu
        F = lam

        # find starting radius
        r_start = start_level(vs, omega, l, r_0, R)

        integrator = ode(dY_dr_homo)
        integrator.set_f_params(A, C, L, N, F, rho, l, omega)
    else:
        raise ValueError('either provide a pymesher model or vs, rho and R')

    # assume homogeneous model below some starting radius
    nr = len(r_in_m)
    Y2 = np.zeros(nr)

    # use analytical solution as initial condition and for radii below the
    # starting radius
    mask = np.logical_and(r_in_m > 0., r_in_m <= r_start)
    Y2[mask] = Y_initial_conditions_new(r_in_m[mask], vp, vs, rho, l, omega)[1]

    if r_start == R or omega == 0.:
        if return_compat_mode:
            return r_in_m, None, Y2, -1, 0
        else:
            return r_in_m, Y2, -1, r_start

    # make sure to smoothly connect to the initial conditian
    if np.any(mask):
        Y2_scale = (
            Y_initial_conditions(r_start, vp, vs, rho, l, omega)[1] /
            Y_initial_conditions_new(r_start, vp, vs, rho, l, omega)[1])

        Y2 *= Y2_scale

    # Y_initial_conditions gives a better order of magnitude, but is not the
    # solution itself as a function of radius but scaled with some function of
    # radius
    initial_conditions = Y_initial_conditions(r_start, vp, vs, rho, l, omega)
    scale = np.max(np.abs(initial_conditions))
    if np.abs(np.array(initial_conditions)).sum() == 0:
        raise InitError('zero initial conditions')

    initial_conditions = np.array(initial_conditions) / scale
    Y2 /= scale

    integrator.set_integrator('dopri5', nsteps=nsteps, rtol=rtol,
                              first_step=(R - r_start) / nsteps / 100)
    integrator.set_initial_value(initial_conditions, r_start)

    # add a zero counter to get the mode count
    mc = mode_counter(integrator.f, integrator.f_params, numerator_idx=(4, 3),
                      denominator_idx=1, ndim=4)
    integrator.set_solout(mc)

    # Do the actual integration
    for i in np.arange(nr - 1):
        if r_in_m[i+1] < r_start:
            continue

        integrator.integrate(r_in_m[i+1])

        if not integrator.successful():
            raise RuntimeError(
                "Integration Error while integrating radial equation")

        Y2[i+1] = integrator.y[1]

        # avoid float overflows by rescaling if the solution grows big
        rfac = 10. ** int(np.log10(abs(Y2[i+1])))

        if rfac > 1.:
            integrator.set_initial_value(integrator.y / rfac, integrator.t)
            Y2 /= rfac

    if return_compat_mode:
        return r_in_m, None, Y2, mc.count, 0
    else:
        return r_in_m, Y2, mc.count, r_start
