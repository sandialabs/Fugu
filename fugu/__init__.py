#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:06:25 2019

@author: smusuva
"""

from fugu import bricks
from fugu import scaffold
from fugu.bricks import Brick

from fugu.scaffold import Scaffold
from fugu.backends import Backend, snn_Backend, ds_Backend, pynn_Backend


input_coding_types = [
                       'current',
                       'unary-B',
                       'unary-L',
                       'binary-B',
                       'binary-L',
                       'temporal-B',
                       'temporal-L',
                       'Raster',
                       'Population',
                       'Rate',
                       'Undefined',
                       ]
