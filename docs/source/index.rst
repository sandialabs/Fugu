.. Fugu documentation master file, created by
   sphinx-quickstart on Thu Mar 31 00:56:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Fugu's documentation!
================================



.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive: 
   :private-members:




.. toctree::
   :maxdepth: 4
   :caption: Contents:


   index
   introduction
   fugu/backends/*.py
   fugu/bricks/*.py
   fugu/scaffold/*.py
   fugu/simulators/SpikingNeuralNetwork/neuralnetwork/*.py
   fugu/simulators/SpikingNeuralNetwork/neuron/*.py
   fugu/simulators/SpikingNeuralNetwork/synapse/*.py
   fugu/utils/*.py
   examples/*.py

   examples.fugu_test
   

Modules
========
.. autosummary::
   :toctree: modules

   fugu
   fugu.backends
   fugu.bricks
   fugu.scaffold
   fugu.simulators.SpikingNeuralNetwork.neuralnetwork
   fugu.simulators.SpikingNeuralNetwork.neuron
   fugu.simulators.SpikingNeuralNetwork.synapse
      
   fugu.utils

   examples




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
