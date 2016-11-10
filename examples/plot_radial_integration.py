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

from pymodes import toroidal
from pymodes import eigenfrequencies


# for homogeneous model, check eigenfrequencies

model = pymesher.model.read('homo_model.bm')

freq = eigenfrequencies.analytical_eigen_frequencies(
    omega_max=0.01, omega_delta=0.00001, l=10, rho=1e3, vs=1e3, vp=1.7e3,
    R=6371e3, mode='T')

for f in freq[:2]:
    print f
    r, y1, y2 = toroidal.integrate_radial(model, l=10, omega=f, r_0=1.,
                                         nsamp_per_layer=100, rtol=1e-10)

    plt.plot(r, y1 / y1.ptp(), 'r')
    plt.plot(r, y2 / y2.ptp(), 'k')


# integrate radially in prem (eigenfrequency not known)

model = pymesher.model.built_in('prem_iso')
r, y1, y2 = toroidal.integrate_radial(model, l=100, omega=0.015 * 2 * np.pi,
                                      r_0=3480e3 + 1e-6, nsamp_per_layer=100)
plt.figure()
plt.plot(r, y1 / y1.ptp(), 'r')
plt.plot(r, y2 / y2.ptp(), 'k')

plt.show()
