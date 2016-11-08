#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""
pymodes: normal mode and surface wave stuff

:copyright:
    Martin van Driel, Martin@vanDriel.de
    Federico Munch
:license:
    None
"""

from setuptools import setup

from numpy.distutils.core import setup, Extension

setup(name='pymodes',
      version='0.1',
      description='normal mode and surface wave stuff',
      author='Martin van Driel',
      author_email='Martin@vanDriel.de',
      license='None',
      packages=['pymodes'],
      zip_safe=False)
