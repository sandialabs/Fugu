import numpy as np

import fugu
from fugu import Scaffold
from fugu.bricks import SynapseProperties
from fugu.backends import ds_Backend, snn_Backend

no_time = Scaffold()

no_time.add_brick(SynapseProperties(weights=[0.5, 0.5, 0.5, 0.5], name='Test'),
                  output=True)

no_time.lay_bricks()
no_time.summary(verbose=2)

graph_names = list(no_time.graph.nodes.data('name'))

params = {}
params['Test'] = {}
params['Test']['weights'] = [1.1, 1.1, 1.0, 2.1]

backend = ds_Backend()
backend.compile(no_time, {})
results = backend.run(10)
for row in results.itertuples():
    neuron_name = graph_names[int(row.neuron_number)][0]
    print(neuron_name, row.time)

backend.reset()
backend.set_parameters(params)
results = backend.run(10)
for row in results.itertuples():
    neuron_name = graph_names[int(row.neuron_number)][0]
    print(neuron_name, row.time)
