#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from .test_core import SnnInstantDecayTests
from .test_core import SnnVectorInputTests
from .test_core import SnnChangeNeuronPropertyTests
from .test_core import SnnChangeSynapseInternalPropertyTests
from .test_core import SnnChangeSynapseExternalPropertyTests
from .test_core import SnnCompoundBrickTests
from .test_utility_bricks import SnnTemporalAdderTests
from .test_register_bricks import SnnAdditionTests
from .test_register_bricks import SnnSubtractionTests
from .test_register_bricks import SnnRegisterTests
from .test_register_bricks import SnnMaxTests
from .test_app_bricks import SnnLISTests
from .test_stochastic_bricks import SnnThresholdTests
from .test_graph_bricks import SnnSimpleGraphTraversalTests
from .test_graph_bricks import SnnRegisterGraphTraversalTests

loader = unittest.TestLoader()

snn_suite = unittest.TestSuite()
snn_suite.addTest(loader.loadTestsFromTestCase(SnnInstantDecayTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnVectorInputTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnCompoundBrickTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnChangeNeuronPropertyTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnChangeSynapseInternalPropertyTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnChangeSynapseExternalPropertyTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnTemporalAdderTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnRegisterTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnAdditionTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnSubtractionTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnMaxTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnLISTests))
#snn_suite.addTest(loader.loadTestsFromTestCase(SnnThresholdTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnSimpleGraphTraversalTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnRegisterGraphTraversalTests))
