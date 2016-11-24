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


def start_level(v, omega, l, r_min, r_max, nsamp=10000, tol_start=15,
                rfac=1.5):

    r = np.linspace(r_min, rfac * r_max, nsamp, endpoint=True)

    if r[0] == 0.:
        r = r[1:]

    if omega == 0.:
        return r[0]

    if callable(v):
        v = np.array(v(np.clip(r, r_min, r_max)))
    else:
        v = float(v)

    pp = (l + 0.5) ** 2 / omega ** 2
    # pp = l * (l + 1.) / omega ** 2

    q = 1. / v ** 2 - pp / r ** 2

    # check if turning point is above r_max
    if q[-1] < 0:
        return r_max

    i_turn = max(np.argmax(q > 0), 0)

    # if turning point below r_min
    if i_turn == 0:
        return r[0]

    # go down until the asymptotic solution has gone down to exp(-tol_start)
    else:
        dr = (r[1:] - r[:-1])[:i_turn][::-1]
        s = omega * np.sqrt(-q[:i_turn][::-1]) * dr

        cs = cumtrapz(s)
        if cs[-1] < tol_start:
            return r[0]
        else:
            j = np.argmax(cs > tol_start)
            return min(r[i_turn - j], r_max)
