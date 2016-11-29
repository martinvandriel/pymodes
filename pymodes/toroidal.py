#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Compute eigenfrequencies of toroidal modes in a homogeneous isotropic elastic
full sphere.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import numpy as np
from scipy.integrate import ode
from scipy.special import spherical_jn

from .start_level import start_level
from .misc import get_radial_sampling, InitError
from .mode_counter import mode_counter


def analytical_bc(omega, l, rho, vs, vp, R):
    """
    Compute the the 'characteristic' function according
    to: Takeuchi & Saito (1972), in 'Methods in computational
    physics, Volume 11', Eq (95)

    :type omega: float, scalar or array like
    :param omega: frequency at which to compute the matrix
    :type l: int
    :param l: angular order
    :type rho: float
    :param rho: density in kg / m ** 3
    :type vs: float
    :param float: s-wave velocity in m / s
    :type R: float
    :param R: radius of the sphere
    """
    mu = rho * vs ** 2
    omega = np.array(omega)
    k = omega / vs

    j_n = spherical_jn(l, k * R)
    j_np1 = spherical_jn(l+1, k * R)

    y2 = mu / R * ((l - 1) * j_n - k * R * j_np1)

    return y2


def analytical_characteristic_function(omega, l, rho, vs, vp, R,
                                       gravity=False):
    """
    Compute the 'characteristic' or function - for toroidal modes this is just
    a wrapper to analytical_bc, see Takeuchi & Saito Eq (77)

    for paramters see analytical_bc
    """

    return analytical_bc(omega=omega, l=l, rho=rho, vs=vs, vp=vp, R=R)


def dy_dr(r, y, model, l, omega):
    """
    Takeuchi & Saito (1972), Eq. (76) for general models.
    """

    if model.anisotropic:
        vsv = model.get_native_parameter('VSV', r/model.scale)
        vsh = model.get_native_parameter('VSH', r/model.scale)
    else:
        vsv = model.get_native_parameter('VS', r/model.scale)
        vsh = vsv

    rho = model.get_native_parameter('RHO', r/model.scale)

    L = rho * vsv ** 2
    N = rho * vsh ** 2

    return dy_dr_homo(r, y, L, N, rho, l, omega)


def dy_dr_homo(r, y, L, N, rho, l, omega):
    """
    Takeuchi & Saito (1972), Eq. (76) for homogeneous models.
    """
    dy1_dr = 1 / r * y[0] + 1 / L * y[1]
    dy2_dr = (((l - 1) * (l + 2) * N / r ** 2 - omega ** 2 * rho) * y[0] -
              3 / r * y[1])

    return [dy1_dr, dy2_dr]


def y_initial_conditions(r1, vs, rho, l, omega):
    """
    Takeuchi & Saito (1972), Eq. (95).
    """
    if omega > 0.:
        mu = rho * vs ** 2
        k = omega / vs

        j_n = spherical_jn(l, k * r1)
        j_np1 = spherical_jn(l+1, k * r1)

        y1 = j_n
        y2 = mu / r1 * ((l - 1) * j_n - k * r1 * j_np1)
    else:
        # trivial solution in case of omega = 0, see Dahlen & Tromp section
        # 8.7.2
        y1 = r1
        y2 = 0.

    return y1, y2


def integrate_radial(omega, l, rho=None, vs=None, R=None, model=None,
                     nsteps=10000, rtol=1e-15, r_0=None, nsamp_per_layer=100,
                     **kwargs):
    """
    integrate Takeuchi & Saito (1972), Eq (76) radially, subject to stress free
    initial conditions (e.g. the CMB).
    """

    if model is not None:
        if not model.get_fluid_regions() == []:
            r_0 = (model.get_fluid_regions()[-1][1] + 1e-6) * model.scale

        elif r_0 is None:
            r_0 = 0.

        r_in_m = get_radial_sampling(model, nsamp_per_layer, r_0)

        # find starting radius
        def vs_func(_r):
            return model.get_elastic_parameter('VSH', _r / model.scale)

        r_start = start_level(vs_func, omega, l, r_0, model.scale)

        # setup parameters for asymptotic solution / initial condition below
        # r_start
        _r = np.array([r_start]) / model.scale
        vs = model.get_elastic_parameter('VSH', _r)[0]
        rho = model.get_elastic_parameter('RHO', _r)[0]
        R = model.scale

        integrator = ode(dy_dr)
        integrator.set_f_params(model, l, omega)

    elif rho is not None and vs is not None and R is not None:
        if r_0 is None:
            r_0 = 0.

        r_in_m = np.linspace(r_0, R, nsamp_per_layer+1)

        L = rho * vs ** 2
        N = rho * vs ** 2

        # find starting radius
        r_start = start_level(vs, omega, l, r_0, R)

        integrator = ode(dy_dr_homo)
        integrator.set_f_params(L, N, rho, l, omega)
    else:
        raise ValueError('either provide a pymesher model or vs, rho and R')

    # prepare storage for the eigenfunctions
    nr = len(r_in_m)
    y1 = np.zeros(nr)
    y2 = np.zeros(nr)

    if r_start == r_0:
        # initial condition is stress free
        y1[0] = 1.
        y2[0] = 0.
        initial_conditions = np.array([1., 0.])

    else:
        # use analytical solution as initial condition and for radii below the
        # starting radius
        mask = np.logical_and(r_in_m > 0., r_in_m <= r_start)
        y1[mask], y2[mask] = y_initial_conditions(
            r_in_m[mask], vs, rho, l, omega)

        initial_conditions = y_initial_conditions(r_start, vs, rho, l, omega)

    if (np.array(initial_conditions) ** 2).sum() == 0:
        raise InitError('zero initial conditions')

    # compute the sign of the secular function at the starting radius. Needed
    # to search zero crossings in the secular function
    dy_dr_start_sign = np.sign(integrator.f(r_start, initial_conditions,
                                            *integrator.f_params)[1])

    if r_start == R:
        return r_in_m, y1, y2, -1, 0

    # set integrator parameters: first_step chosen conservative to help the
    # addaptive step size algorithm get started
    integrator.set_integrator('dopri5', nsteps=nsteps, rtol=rtol,
                              first_step=(R - r_start) / nsteps / 100)
    # beta=0.15, safety=0.9, dfactor=0.2)

    integrator.set_initial_value(initial_conditions, r_start)

    # add a zero counter to get the mode count
    mc = mode_counter(integrator.f, integrator.f_params, numerator_idx=0,
                      denominator_idx=1, ndim=2, displacement_idx=0)
    integrator.set_solout(mc)

    # Do the actual integration
    for i in np.arange(nr - 1):
        if r_in_m[i+1] < r_start:
            continue

        integrator.integrate(r_in_m[i+1])

        if not integrator.successful():
            raise RuntimeError(
                "Integration Error while intgrating radial equation")

        y1[i+1], y2[i+1] = integrator.y

        # avoid float overflows by rescaling if the solution grows big
        rfac = 10. ** int(np.log10(np.max(np.abs([y1[i+1], y2[i+1]]))))
        if rfac > 1.:
            integrator.set_initial_value(integrator.y / rfac, integrator.t)
            y1 /= rfac
            y2 /= rfac

    # mode count is the zero crossings as counted by zero_counter + special
    # case of the first modes that have positive derivative of the traction
    # (see Dahlen & Tromp Figure 8.6).
    mode_count = mc.count + (dy_dr_start_sign < 0)
    n = mc.count2
    return r_in_m, y1, y2, mode_count, n
