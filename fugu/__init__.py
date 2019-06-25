#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:06:25 2019

@author: smusuva
"""

#from Fugu.code import bricks as bricks
#from Fugu.code import scaffold

from fugu import code
from fugu.code import bricks
from fugu.code import scaffold
from fugu.code.bricks import Brick

#from Fugu.code.bricks import Brick, InputBrick, Vector_Input, Spike_Input, PRN
#from Fugu.code.bricks import Threshold, Dot, Copy, Concatenate, AND_OR
#from Fugu.code.bricks import Shortest_Path, Breadth_First_Search
#from Fugu.code.bricks import LongestIncreasingSubsequence
#from Fugu.code.bricks import ParityCheck
from fugu.code.scaffold import Scaffold
from fugu.code import backends

#import Fugu.code.bricks as bricks
#import Fugu.code.scaffold as scaffold
#from Fugu.code.scaffold import scaffold

input_coding_types = ['current',
                      'unary-B',
                      'unary-L',
                      'binary-B',
                      'binary-L',
                     'temporal-B',
                     'temporal-L',
                      'Raster',
                      'Population',
                      'Rate',
                     'Undefined']