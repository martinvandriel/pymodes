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
from scipy.integrate import ode, cumtrapz
from scipy.special import spherical_jn


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
    mu = rho * vs ** 2
    k = omega / vs

    j_n = spherical_jn(l, k * r1)
    j_np1 = spherical_jn(l+1, k * r1)

    y1 = j_n
    y2 = mu / r1 * ((l - 1) * j_n - k * r1 * j_np1)

    return y1, y2


class IntegrationOverflow(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class InitError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class TurningPointError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class zero_counter(object):
    """
    a helper class to detect zero crossing in integration with
    scipy.integrate.ode and count the number of steps. To be attached to the
    ode integrator with integrator.set_solout().
    """

    def __init__(self, dy_dt, dy_dt_args):
        self.previous = 0.
        self.count = 0
        self.nstep = 0
        self.dy_dt = dy_dt
        self.dy_dt_args = dy_dt_args

    def __call__(self, t, y):
        self.nstep += 1

        # if np.max(np.abs(y)) > 0 and int(np.log10(np.max(np.abs(y)))) > 100:
        #     raise IntegrationOverflow('')

        # use xor instead of multiplication to avoid float overflows
        if ((self.previous > 0) and (y[1] < 0) or
           (self.previous < 0) and (y[1] > 0)):
            # see Al-Attar MsC Thesis, 2007, eq C.166
            dy_dt = self.dy_dt(t, y, *self.dy_dt_args)
            self.count += -int(np.sign(y[0]) * np.sign(dy_dt[1]))

        self.previous = y[1]


def integrate_radial(omega, l, rho=None, vs=None, R=None, model=None,
                     nsteps=10000, rtol=1e-15, r_0=None, nsamp_per_layer=100):
    """
    integrate Takeuchi & Saito (1972), Eq (76) radially, subject to stress free
    initial conditions (e.g. the CMB).
    """

    if model is not None:
        if not model.get_fluid_regions() == []:
            r_0 = model.get_fluid_regions()[-1][1] * model.scale + 1e-3

        elif r_0 is None:
            r_0 = 0.55e-6 * model.scale

        # adapt discontinuities to r_0
        idx = model.discontinuities > r_0 / model.scale
        ndisc = idx.sum() + 1

        discontinuities = np.zeros(ndisc)
        discontinuities[0] = r_0 / model.scale
        discontinuities[1:] = model.discontinuities[idx]

        # build sampling for return arrays
        r = np.concatenate([np.linspace(discontinuities[iregion],
                                        discontinuities[iregion+1],
                                        nsamp_per_layer, endpoint=False)
                            for iregion in range(ndisc-1)])
        r = np.r_[r, np.array([1.])]
        r_in_m = r * model.scale

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
            r_0 = 0.55e-6 * R

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
        y1[r_in_m < r_start], y2[r_in_m < r_start] = y_initial_conditions(
            r_in_m[r_in_m < r_start], vs, rho, l, omega)

        initial_conditions = y_initial_conditions(r_start, vs, rho, l, omega)

    if (np.array(initial_conditions) ** 2).sum() == 0:
        raise InitError('zero initial conditions')

    # compute the sign of the secular function at the starting radius. Needed
    # to search zero crossings in the secular function
    dy_dr_start_sign = np.sign(integrator.f(r_start, initial_conditions,
                                            *integrator.f_params)[1])

    # set integrator parameters: first_step chosen conservative to help the
    # addaptive step size algorithm get started
    integrator.set_integrator('dopri5', nsteps=nsteps, rtol=rtol,
                              first_step=(R - r_start) / nsteps / 100)
    # beta=0.15, safety=0.9, dfactor=0.2)

    integrator.set_initial_value(initial_conditions, r_start)

    # add a zero counter to get the mode count
    zc = zero_counter(integrator.f, integrator.f_params)
    integrator.set_solout(zc)

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
        while abs(y1[i+1]) > 1e3 or abs(y2[i+1]) > 1e3:
            y1 /= 10
            y2 /= 10

            integrator.set_initial_value([y1[i+1], y2[i+1]], integrator.t)

    return r_in_m, y1, y2, zc.count, dy_dr_start_sign, r_start


def start_level(vs, omega, l, r_min, r_max, nsamp=10000, tol_start=15):

    r = np.linspace(r_min, r_max, nsamp, endpoint=False)

    if omega == 0.:
        return r[-1]

    if callable(vs):
        v = np.array(vs(r))
    else:
        v = float(vs)

    pp = (l + 0.5) ** 2 / omega ** 2

    q = 1. / v ** 2 - pp / r ** 2

    # check if turning point is above r_max
    if q[-1] < 0:
        raise TurningPointError('turning point above r_max')

    i_turn = max(np.argmax(q > 0), 0)

    # if turning point below r_min
    if i_turn == 0:
        return r_min

    # go down until the asymptotic solution has gone down to exp(-tol_start)
    else:
        dr = (r[1:] - r[:-1])[:i_turn][::-1]
        s = omega * np.sqrt(-q[:i_turn][::-1]) * dr

        cs = cumtrapz(s)
        if cs[-1] < tol_start:
            return r_min
        else:
            j = np.argmax(cs > tol_start)
            return r[i_turn - j]
