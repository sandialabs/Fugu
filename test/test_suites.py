import unittest

from .test_core import SnnInstantDecayTests, DsInstantDecayTests
from .test_core import PynnSpinnakerInstantDecayTests, PynnBrianInstantDecayTests
from .test_core import SnnVectorInputTests, DsVectorInputTests
from .test_core import PynnSpinnakerVectorInputTests, PynnBrianVectorInputTests
from .test_core import SnnChangeNeuronPropertyTests, DsChangeNeuronPropertyTests
from .test_core import PynnSpinnakerChangeNeuronPropertyTests, PynnBrianChangeNeuronPropertyTests
from .test_core import SnnChangeSynapseInternalPropertyTests, DsChangeSynapseInternalPropertyTests
from .test_core import PynnSpinnakerChangeSynapseInternalPropertyTests, PynnBrianChangeSynapseInternalPropertyTests
from .test_core import SnnChangeSynapseExternalPropertyTests, DsChangeSynapseExternalPropertyTests
from .test_core import PynnSpinnakerChangeSynapseExternalPropertyTests, PynnBrianChangeSynapseExternalPropertyTests
from .test_core import SnnCompoundBrickTests, DsCompoundBrickTests
from .test_core import PynnSpinnakerCompoundBrickTests, PynnBrianCompoundBrickTests
from .test_utility_bricks import SnnTemporalAdderTests, DsTemporalAdderTests
from .test_utility_bricks import PynnSpinnakerTemporalAdderTests, PynnBrianTemporalAdderTests
from .test_register_bricks import SnnAdditionTests, DsAdditionTests
from .test_register_bricks import PynnSpinnakerAdditionTests, PynnBrianAdditionTests
from .test_register_bricks import SnnSubtractionTests, DsSubtractionTests
from .test_register_bricks import PynnSpinnakerSubtractionTests, PynnBrianSubtractionTests
from .test_register_bricks import SnnRegisterTests, DsRegisterTests
from .test_register_bricks import PynnSpinnakerRegisterTests, PynnBrianRegisterTests
from .test_register_bricks import SnnMaxTests, DsMaxTests
from .test_register_bricks import PynnSpinnakerMaxTests, PynnBrianMaxTests
from .test_app_bricks import SnnLISTests, DsLISTests
from .test_app_bricks import PynnSpinnakerLISTests, PynnBrianLISTests
from .test_stochastic_bricks import SnnThresholdTests, DsThresholdTests
from .test_graph_bricks import SnnSimpleGraphTraversalTests, DsSimpleGraphTraversalTests
from .test_graph_bricks import PynnSpinnakerSimpleGraphTraversalTests, PynnBrianSimpleGraphTraversalTests
from .test_graph_bricks import SnnRegisterGraphTraversalTests, DsRegisterGraphTraversalTests
from .test_graph_bricks import PynnSpinnakerRegisterGraphTraversalTests, PynnBrianRegisterGraphTraversalTests

loader = unittest.TestLoader()

debug_suite = unittest.TestSuite()
#debug_suite.addTest(loader.loadTestsFromTestCase(SnnSimpleGraphTraversalTests))
debug_suite.addTest(loader.loadTestsFromTestCase(DsSimpleGraphTraversalTests))
#debug_suite.addTest(loader.loadTestsFromTestCase(SnnRegisterGraphTraversalTests))
#debug_suite.addTest(loader.loadTestsFromTestCase(SnnRegisterTests))
#debug_suite.addTest(loader.loadTestsFromTestCase(DsRegisterGraphTraversalTests))
#debug_suite.addTest(loader.loadTestsFromTestCase(DsRegisterTests))

debug2_suite = unittest.TestSuite()
#debug2_suite.addTest(loader.loadTestsFromTestCase(PynnBrianSimpleGraphTraversalTests))
#debug2_suite.addTest(loader.loadTestsFromTestCase(PynnBrianRegisterGraphTraversalTests))
#debug2_suite.addTest(loader.loadTestsFromTestCase(PynnBrianRegisterTests))

snn_suite = unittest.TestSuite()
snn_suite.addTest(loader.loadTestsFromTestCase(SnnInstantDecayTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnVectorInputTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnCompoundBrickTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnChangeNeuronPropertyTests))
snn_suite.addTest(
    loader.loadTestsFromTestCase(SnnChangeSynapseInternalPropertyTests))
snn_suite.addTest(
    loader.loadTestsFromTestCase(SnnChangeSynapseExternalPropertyTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnTemporalAdderTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnRegisterTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnAdditionTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnSubtractionTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnMaxTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnLISTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnThresholdTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnSimpleGraphTraversalTests))
snn_suite.addTest(loader.loadTestsFromTestCase(SnnRegisterGraphTraversalTests))

ds_suite = unittest.TestSuite()
ds_suite.addTest(loader.loadTestsFromTestCase(DsInstantDecayTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsVectorInputTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsCompoundBrickTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsChangeNeuronPropertyTests))
ds_suite.addTest(
    loader.loadTestsFromTestCase(DsChangeSynapseInternalPropertyTests))
ds_suite.addTest(
    loader.loadTestsFromTestCase(DsChangeSynapseExternalPropertyTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsTemporalAdderTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsRegisterTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsAdditionTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsSubtractionTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsMaxTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsLISTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsThresholdTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsSimpleGraphTraversalTests))
ds_suite.addTest(loader.loadTestsFromTestCase(DsRegisterGraphTraversalTests))

pynn_spinnaker_suite = unittest.TestSuite()
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerInstantDecayTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerVectorInputTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerCompoundBrickTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerChangeNeuronPropertyTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerTemporalAdderTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerRegisterTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerAdditionTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerSubtractionTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerMaxTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerLISTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerSimpleGraphTraversalTests))
pynn_spinnaker_suite.addTest(
    loader.loadTestsFromTestCase(PynnSpinnakerRegisterGraphTraversalTests))
# NOTE: Spinnaker does not support the functionality for these tests
#pynn_spinnaker_suite.addTest(loader.loadTestsFromTestCase(PynnSpinnakerChangeSynapseInternalPropertyTests))
#pynn_spinnaker_suite.addTest(loader.loadTestsFromTestCase(PynnSpinnakerChangeSynapseExternalPropertyTests))

pynn_brian_suite = unittest.TestSuite()
pynn_brian_suite.addTest(
    loader.loadTestsFromTestCase(PynnBrianInstantDecayTests))
pynn_brian_suite.addTest(
    loader.loadTestsFromTestCase(PynnBrianVectorInputTests))
pynn_brian_suite.addTest(
    loader.loadTestsFromTestCase(PynnBrianCompoundBrickTests))
pynn_brian_suite.addTest(
    loader.loadTestsFromTestCase(PynnBrianChangeNeuronPropertyTests))
pynn_brian_suite.addTest(
    loader.loadTestsFromTestCase(PynnBrianChangeSynapseInternalPropertyTests))
pynn_brian_suite.addTest(
    loader.loadTestsFromTestCase(PynnBrianChangeSynapseExternalPropertyTests))
pynn_brian_suite.addTest(
    loader.loadTestsFromTestCase(PynnBrianTemporalAdderTests))
pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianRegisterTests))
pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianAdditionTests))
pynn_brian_suite.addTest(
    loader.loadTestsFromTestCase(PynnBrianSubtractionTests))
pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianMaxTests))
pynn_brian_suite.addTest(loader.loadTestsFromTestCase(PynnBrianLISTests))
pynn_brian_suite.addTest(
    loader.loadTestsFromTestCase(PynnBrianSimpleGraphTraversalTests))
pynn_brian_suite.addTest(
    loader.loadTestsFromTestCase(PynnBrianRegisterGraphTraversalTests))
