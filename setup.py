#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

package_list = find_packages()

setup(name='fugu',
      version='0.1',
      description='A python library for computational neural graphs',
      install_requires = ['decorator',
                  'future',
                  'greenlet',
                  'msgpack',
                  'pandas>1.0.4',
                  'python-dateutil',
                  'pytz',
                  'networkx==2.4',
                  'numpy~=1.19.0',
                  'six==1.15.0',
                  'unittest'        ],
      packages=package_list)
