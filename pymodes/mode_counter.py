#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A helper class to do mode counting.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import numpy as np


class mode_counter(object):
    """
    a helper class to detect zero crossing in integration with
    scipy.integrate.ode and count the number of steps. To be attached to the
    ode integrator with integrator.set_solout().
    """

    def __init__(self, dy_dt, dy_dt_args, numerator_idx=0, denominator_idx=1,
                 ndim=2, displacement_idx=None):
        self.dy_dt = dy_dt
        self.dy_dt_args = dy_dt_args
        self.numerator_idx = numerator_idx
        self.denominator_idx = denominator_idx
        self.ndim = ndim
        self.displacement_idx = displacement_idx

        self.previous = np.zeros(ndim)
        self.count = 0
        self.count2 = 0 if displacement_idx is not None else None
        self.nstep = 0

    def __call__(self, t, y):
        self.nstep += 1

        if self.previous[self.denominator_idx] * y[self.denominator_idx] < 0.:
            # see Al-Attar MsC Thesis, 2007, eq C.166
            dy_dt = self.dy_dt(t, y, *self.dy_dt_args)
            if isinstance(self.numerator_idx, (int, long)):
                self.count += -int(np.sign(y[self.numerator_idx]) *
                                   np.sign(dy_dt[self.denominator_idx]))
            elif type(self.numerator_idx) is tuple:
                self.count += int(np.sign(y[self.numerator_idx[0]] -
                                          y[self.numerator_idx[1]]) *
                                  np.sign(dy_dt[self.denominator_idx]))

        # count zero crossings of the displacement
        if self.displacement_idx is not None:
            if (self.previous[self.displacement_idx] *
               y[self.displacement_idx]) < 0:
                self.count2 += 1

        self.previous = np.array(y)
