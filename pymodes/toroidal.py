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


class zero_counter():
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
        if self.previous * y[1] < 0.:
            # see Al-Attar MsC Thesis, 2007, eq C.166
            dy_dt = self.dy_dt(t, y, *self.dy_dt_args)
            self.count += -int(np.sign(y[0]) * np.sign(dy_dt[1]))
        self.previous = y[1]


def integrate_radial(omega, l, rho=None, vs=None, R=None, model=None,
                     nsteps=10000, rtol=1e-15, r_0=1e-10, nsamp_per_layer=100):
    """
    integrate Takeuchi & Saito (1972), Eq (76) radially, subject to stress free
    initial conditions (e.g. the CMB).
    """

    if model is not None:
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

        integrator = ode(dy_dr)
        integrator.set_f_params(model, l, omega)

    elif rho is not None and vs is not None and R is not None:
        r_in_m = np.linspace(r_0, R, nsamp_per_layer+1)

        L = rho * vs ** 2
        N = rho * vs ** 2

        integrator = ode(dy_dr_homo)
        integrator.set_f_params(L, N, rho, l, omega)
    else:
        raise ValueError('either provide a pymesher model or vs, rho and R')

    # assume to start at a stress free boundary and set initial conditions
    nr = len(r_in_m)
    y1 = np.zeros(nr)
    y1[0] = 1.
    y2 = np.zeros(nr)
    y2[0] = 0.

    integrator.set_integrator('dopri5', nsteps=nsteps, rtol=rtol)
    integrator.set_initial_value([y1[0], y2[0]], r_0)

    zc = zero_counter(integrator.f, integrator.f_params)
    integrator.set_solout(zc)

    # Do the actual integration
    for i in np.arange(nr - 1):

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

    return r_in_m, y1, y2, zc.count
