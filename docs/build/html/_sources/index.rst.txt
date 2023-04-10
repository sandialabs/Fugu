.. Fugu documentation master file, created by
   sphinx-quickstart on Thu Mar 31 00:56:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================
Fugu
===================

Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Overview 
================================
A python library for computational neural graphs.

Fugu is a high-level framework specifically designed for developing spiking circuits in terms of computation graphs. Accordingly, with a base leaky-integrate-and fire (LIF) neuron model at its core, neural circuits are built as *bricks*. These foundational computations are then combined and composed as scaffolds to construct larger computations. This allows us to describe spiking circuits in terms of neural features common to most NMC architectures rather than platform specific designs. In addition to architectural abstraction, the compositionality concept of Fugu not only facilitates a hierarchical approach to functionality development but also enables adding pre and post processing operations to overarching neural circuits. Such properties position Fugu to help explore under what parameterization or scale a neural approach may offer an advantage. For example, prior work has analyzed neural algorithms for computational kernels like sorting, optimization, and graph analytics identifying different regions in which a neural advantage exists accounting for neural circuit setup, timing, or other factors.

Fugu is open source and availble on github:  `<https://github.com/SNL-NERL/Fugu>`_

Install
================================

Dependencies
________________________________
A full list of dependencies is listed in requirements.txt.  The high level dependencies are:

- Numpy
- NetworkX
- Pandas

A Note on Running Examples
________________________________

Some of the examples additionally require Jupyter and matplotlib.


Install using Pip:

.. code-block::

   git clone http://github.com/SNL-NERL/Fugu.git
   cd Fugu
   pip install -r requirements.txt
   pip install --upgrade .

Documentation
================================

Documentation is available in a few different places.

A good place to start is the examples folder on github: `<http://github.com/SNL-NERL/Fugu/tree/main/examples>`_

The neuron model is explained `here. <http://snl-nerl.github.io/Fugu/build/html/neuron_model.html>`_

Links to the API are below.

.. toctree::
   :maxdepth: 3
   :caption: API Docs
   
   fugu/index


Collaborations and Contributing
================================
We hope that as Fugu develops it will become a useful tool for the spiking neuromorphic field. 

While the Fugu team is small, we are always interested in collaborations where possible.  If you have questions 
or are interested in starting a collaboration, please e-mail us at `wg-fugu@sandia.gov <wg-fugu@sandia.gov>`_. 

If you are not a collaborator but would like to contribute, you are welcome to create an github issue or e-mail to discuss.  
In many cases, extensions to Fugu should not be incoporated into the core of Fugu, see `<http://snl-nerl.github.io/Fugu/build/html/contributing.html>`_.

If you use Fugu for your research, please cite the following paper:

Aimone, James B., William Severa, and Craig M. Vineyard. "Composing neural algorithms with Fugu." Proceedings of the International Conference on Neuromorphic Systems. 2019.


==================
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
