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
from scipy.optimize import brentq

from pymodes.eigenfrequencies import analytical_eigen_frequencies_branch
from pymodes import toroidal
from pymodes import spheroidal


omega_max = 0.01
omega_delta = 0.0000001
omega = np.arange(0., omega_max, omega_delta)

l = 10
rho = 1e3
vs = 1e3
vp = 2e3
R = 6371e3

def plot(det, freq, title=''):
    plt.figure()
    plt.plot(omega, det)
    plt.axhline(0., color='k')
    plt.title(title)

    for f in freq:
        plt.axvline(f, color='k')

det_t = toroidal.analytical_characteristic_function(omega, l, rho, vs, vp, R)

freq_t = analytical_eigen_frequencies_branch(
    omega_max, omega_delta, l, rho, vs, vp, R, mode='T')

plot(det_t, freq_t, 'toroidal')

det_s = spheroidal.analytical_characteristic_function(omega, l, rho, vs, vp, R)

freq_s = analytical_eigen_frequencies_branch(
    omega_max, omega_delta, l, rho, vs, vp, R, mode='S')

plot(det_s, freq_s, 'spheroidal')

print freq_s
f0 = brentq(
    f=spheroidal._k1_sqr, a=0., b=omega_max, args=(l, rho, vs, vp, R),
    xtol=1e-8, maxiter=100,  disp=True)
print f0

plt.axvline(f0, color='r')

k1 = spheroidal._k1_sqr(omega, l, rho, vs, vp, R)
plt.figure()
plt.plot(omega, k1)
plt.axvline(f0, color='r')

plt.show()
