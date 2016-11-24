#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Some helper functions

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import numpy as np


def get_radial_sampling(model, nsamp_per_layer, r_0):
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
    return r * model.scale


class InitError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
