import unittest

from .test_core import SnnInstantDecayTests, DsInstantDecayTests, PynnSpinnakerInstantDecayTests
from .test_core import SnnVectorInputTests, DsVectorInputTests, PynnSpinnakerVectorInputTests
from .test_utility_bricks import SnnAdderTests, DsAdderTests, PynnSpinnakerAdderTests
from .test_utility_bricks import SnnRegisterTests, DsRegisterTests, PynnSpinnakerRegisterTests
from .test_app_bricks import SnnLISTests, DsLISTests, PynnSpinnakerLISTests
from .test_stochastic_bricks import SnnThresholdTests, DsThresholdTests
#from test_graph_bricks import SnnGraphTests, DsGraphTests, PynnSpinnakerGraphTests

# Until pynn releases a brian2 backend, these tests are disabled for now
#from .test_core import PynnBrianInstantDecayTests
#from .test_utility_bricks import PynnBrianAdderTests, PynnBrianRegisterTests
#from .test_app_bricks import PynnBrianLISTests
#from .test_graph_bricks import PynnBrianGraphTests,


loader = unittest.TestLoader()

snn_suite = unittest.TestSuite()
snn_suite.addTest(loader.loadTestsFromTestCase(SnnInstantDecayTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnVectorInputTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnAdderTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnRegisterTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnLISTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnThresholdTests))

ds_suite = unittest.TestSuite()
ds_suite.addTest(loader.loadTestsFromTestCase(DsInstantDecayTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsVectorInputTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsAdderTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsRegisterTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsLISTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsThresholdTests))

#pynn_brian_suite = unittest.TestSuite()
#pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianInstantDecayTests))
#pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianVectorInputTests))
#pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianAdderTests))
#pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianRegisterTests))
#pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianLISTests))

pynn_spinnaker_suite = unittest.TestSuite()
pynn_spinnaker_suite.addTest(loader.loadTestsFromTestCase(PynnSpinnakerInstantDecayTests))
pynn_spinnaker_suite.addTest(loader.loadTestsFromTestCase(PynnSpinnakerVectorInputTests))
pynn_spinnaker_suite.addTest(loader.loadTestsFromTestCase(PynnSpinnakerAdderTests))
pynn_spinnaker_suite.addTest(loader.loadTestsFromTestCase(PynnSpinnakerRegisterTests))
pynn_spinnaker_suite.addTest(loader.loadTestsFromTestCase(PynnSpinnakerLISTests))
