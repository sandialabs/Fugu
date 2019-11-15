import unittest

from test_utility_bricks import SnnUtilityTests, DsUtilityTests, PynnBrianUtilityTests, PynnSpinnakerUtilityTests
from test_stochastic_bricks import SnnStochasticTests, DsStochasticTests
from test_graph_bricks import SnnGraphTests, DsGraphTests, PynnBrianGraphTests, PynnSpinnakerGraphTests
from test_app_bricks import SnnAppTests, DsAppTests, PynnBrianAppTests, PynnSpinnakerAppTests

loader = unittest.TestLoader()

snn_suite = unittest.TestSuite()
#snn_suite.addTest(loader.loadTestsFromTestCase(SnnUtilityTests))
#snn_suite.addTest(loader.loadTestsFromTestCase(SnnStochasticTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnGraphTests))
#snn_suite.addTest(loader.loadTestsFromTestCase(SnnAppTests))

ds_suite = unittest.TestSuite()
#ds_suite.addTest(loader.loadTestsFromTestCase(DsUtilityTests))
#ds_suite.addTest(loader.loadTestsFromTestCase(DsStochasticTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsGraphTests))
#ds_suite.addTest(loader.loadTestsFromTestCase(DsAppTests))

pynn_brian_suite = unittest.TestSuite()
#pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianUtilityTests))
pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianGraphTests))
#pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianAppTests))

pynn_spinnaker_suite = unittest.TestSuite()
pynn_spinnaker_suite.addTest(loader.loadTestsFromTestCase(PynnSpinnakerUtilityTests))
pynn_spinnaker_suite.addTest(loader.loadTestsFromTestCase(PynnSpinnakerGraphTests))
pynn_spinnaker_suite.addTest(loader.loadTestsFromTestCase(PynnSpinnakerAppTests))
