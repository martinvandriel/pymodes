#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A new python script.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import matplotlib.pyplot as plt
import numpy as np
import pymesher

from pymodes import spheroidal
from pymodes import eigenfrequencies


# for homogeneous model, check eigenfrequencies

freq = eigenfrequencies.analytical_eigen_frequencies(
    omega_max=0.01, omega_delta=0.00001, l=10, rho=1e3, vs=1e3, vp=1.7e3,
    R=6371e3, mode='S', gravity=False)

for f in freq[:5]:
    print f
    r, Y2 = spheroidal.integrate_radial(
        rho=1e3, vs=1e3, vp=1.7e3, R=6371e3, l=10, omega=f, r_0=1.,
        nsamp_per_layer=1000, rtol=1e-10, nsteps=10000)

    print Y2[-1] / Y2.ptp()
    plt.plot(r, Y2, 'r')

plt.show()
