import os
import sys
import unittest

parentdir = os.path.abspath('../../')
print("parentdir for this code on your computer is =  " + parentdir)
sys.path.append(parentdir)

import Fugu.fugu.bricks as BRICKS
from Fugu.fugu.backends import snn_Backend
from Fugu.test.base import BrickTest
from Fugu.fugu import *




from .test_instant_decay import SnnInstantDecayTests
from .test_synapse_properties import SnnThresholdTests


# from .test_instant_decay import PynnSpinnakerInstantDecayTests, PynnBrianInstantDecayTests
# from .test_vector_input import SnnVectorInputTests, DsVectorInputTests
# from .test_vector_input import PynnSpinnakerVectorInputTests, PynnBrianVectorInputTests
# from .test_neuron_props import SnnChangeNeuronPropertyTests, DsChangeNeuronPropertyTests
# from .test_neuron_props import PynnSpinnakerChangeNeuronPropertyTests, PynnBrianChangeNeuronPropertyTests
# from .test_internal_synapse_props import SnnChangeSynapseInternalPropertyTests, DsChangeSynapseInternalPropertyTests
# from .test_internal_synapse_props import PynnSpinnakerChangeSynapseInternalPropertyTests, PynnBrianChangeSynapseInternalPropertyTests
# from .test_external_synapse_props import SnnChangeSynapseExternalPropertyTests, DsChangeSynapseExternalPropertyTests
# from .test_external_synapse_props import PynnSpinnakerChangeSynapseExternalPropertyTests, PynnBrianChangeSynapseExternalPropertyTests
# from .test_compound_brick import SnnCompoundBrickTests, DsCompoundBrickTests
# from .test_compound_brick import PynnSpinnakerCompoundBrickTests, PynnBrianCompoundBrickTests
# class SnnInstantDecayTests:
#     pass

