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

from pymodes.eigenfrequencies import analytical_eigen_frequencies
from pymodes import toroidal
from pymodes import spheroidal


omega_max = 0.01
omega_delta = 0.000001
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
    plt.xlim(0., omega_max)

    for f in freq:
        plt.axvline(f, color='k')

det_t = toroidal.analytical_characteristic_function(omega, l, rho, vs, vp, R)

freq_t = analytical_eigen_frequencies(
    omega_max, omega_delta, l, rho, vs, vp, R, mode='T')

plot(det_t, freq_t, 'toroidal')

det_s = spheroidal.analytical_characteristic_function(omega, l, rho, vs, vp, R)

freq_s = analytical_eigen_frequencies(
    omega_max, omega_delta, l, rho, vs, vp, R, mode='S')

plot(det_s, freq_s, 'spheroidal')

plt.show()
