#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:56:33 2019

@author: smusuva
"""
from setuptools import setup, find_packages

package_list = find_packages()

setup(name='fugu',
      version='0.1',
      description='A python library for computational neural graphs',
      install_requires=package_list)
