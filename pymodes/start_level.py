#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Compute the start level for radial integration of normal modes. Adapted from
David Al-Attars yspec package.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import numpy as np
from scipy.integrate import cumtrapz


def start_level(vs, omega, l, r_min, r_max, nsamp=10000, tol_start=15):

    r = np.linspace(r_min, r_max, nsamp, endpoint=False)

    if omega == 0.:
        return r_min

    if callable(vs):
        v = np.array(vs(r))
    else:
        v = float(vs)

    pp = (l + 0.5) ** 2 / omega ** 2

    q = 1. / v ** 2 - pp / r ** 2

    # check if turning point is above r_max
    if q[-1] < 0:
        return r_min

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
