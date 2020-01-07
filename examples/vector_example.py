import numpy as np

import fugu
from fugu import Scaffold
from fugu.bricks import Vector_Input, Threshold, Dot
from fugu.backends import ds_Backend, snn_Backend

print("Without time dimension")
no_time = Scaffold()
spike_times = [0,1,0,1]
vector_1 = Vector_Input(spike_times, coding='Raster', name='input1')
dot_brick = Dot([1.0, 1.0, 1.0, 1.0], name='Dot')
no_time.add_brick(vector_1, 'input')
no_time.add_brick(dot_brick, input_nodes=(0,0))
no_time.add_brick(Threshold(3.0, name='Thresh', output_coding='temporal-L'), input_nodes=(1,0))

no_time.lay_bricks()
graph_names = list(no_time.graph.nodes.data('name'))

params = {}

params['Dot'] = {}
params['Dot']['weight'] = [1.0, 1.0, 1.0, 2.1]

params['compile_args'] = {'record':'all'}

backend = ds_Backend()
backend.compile(no_time, {'record':'all'})
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
