#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A new python script.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import inspect
import numpy as np
import os
import pymesher

from ..start_level import start_level


# Most generic way to get the data directory.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


def test_start_level():
    l = 10
    vs = 1e3
    R = 6371e3
    ri = 1e-10

    n = 10
    omega = np.linspace(0.01, 0.1, n)

    s_ref = np.array([187307.4, 94290.8, 63072.9, 47782.5, 38226., 31855.,
                      27395.3, 24209.8, 21661.4, 19750.1])

    s = np.zeros(n)
    for i, o in enumerate(omega):
        s[i] = start_level(vs, o, l, ri, R, nsamp=10001, rfac=1.)

    np.testing.assert_allclose(s, s_ref, rtol=1e-8)

    model = pymesher.model.read(os.path.join(DATA_DIR, 'homo_model.bm'))

    def vs(_r):
        return model.get_elastic_parameter('VSH', _r / model.scale)

    s = np.zeros(n)
    for i, o in enumerate(omega):
        s[i] = start_level(vs, o, l, ri, R, nsamp=10001, rfac=1.)

    np.testing.assert_allclose(s, s_ref, rtol=1e-8)
