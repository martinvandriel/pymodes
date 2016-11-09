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

from pymodes.eigenfrequencies import analytical_eigen_frequencies_catalogue
from pymodes import spheroidal

omega_max = 0.01
omega_delta = 0.000001
omega = np.arange(0., omega_max, omega_delta)

lmax = 50
rho = 1e3
vs = 1e3
vp = 1.7e3
R = 6371e3

def plot(cat, title=''):
    plt.figure()
    for branch in cat.T:
        plt.plot(branch, 'ro')


cat_s = analytical_eigen_frequencies_catalogue(
    omega_max, omega_delta, lmax, rho, vs, vp, R, mode='S', gravity=True)

plot(cat_s, 'spheroidal')

cat_t = analytical_eigen_frequencies_catalogue(
    omega_max, omega_delta, lmax, rho, vs, vp, R, mode='T')

plot(cat_t, 'toroidal')

plt.show()
