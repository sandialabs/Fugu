## Fugu Neural Networks

### Specifying the network in Fugu
To prepare a network, fugu "bricks" are assembled into a "scaffold".  Each brick may be thought of as generating a graph component analagous to a local neural "circuit" with neurons and synapses (as defined by nodes and edges within the brick).  The scaffold then defines the connectivity between these local neural circuits.  As with any neural circuit or graph, the functional properties of a brick's circuitry are determined by the specific properties of the neurons and synapses as well as the circuit connectivity.  Below we describe the model properties of the neurons and synapses currently available within Fugu.  

It should be noted that each brick is the algorithm for generating a neural circuit, not the graph itself, and that the network object is not constructed until `lay_bricks()` is called.  At evaluation this network is converted into torch.Tensor objects for simulation.  


### Neuron Model
The Fugu neuron model is based upon a standard leaky-integrate-and-fire neuron.  The fugu neuron is constructed with the variables (for neuron $i$):

Variable Key | Definition/Description | Variable Name | Type | 
| ----- | ----- | ----- | ---- |
| potential | Internal state of the neuron, analagous to the biological membrane potential | x_i | float |
| threshold | Neuron spike threshold | T_i | float |
| decay | Decay constant | m_i | float $\in [0,1]$|
| p | Probability of spike (given that $x_i > T_i$) | p_i | float $\in [0,1]$| |
| record | List of values to record | N/A | list of strings |
| N/A | Random Draw | a | float $\in [0,1]$ |


For each simulation step $t$, $x_i$ is computed as follows: 

1. $x_i = x_{t-1,i} + I_i + W_{i}*S_{t-1}$,  
where $I_i$ is the external current injection to neuron $i$, $W_{i}$ is a vector containing the weights of the edges between every other neuron in the network and neuron $i$, and $S_{t-1}$ is a vector containing the spike history of the network for the previous stimulation step.   

2. If $x_i > T_i$, then if $a < p_i$, then $S[i]$  
If the internal state of neuron $i$ is greater than its spike threshold, spike with probability $p_i$.  Currently the internal state is reset to $0$ following a spike.  

3. If $x_i <= T_i$, then $x_i = x_i * (1-m_i)$ 
If neuron $i$ does not spike, the internal state decays to $0$ as described above.  Note: if $m_i = 0$ there is no decay and in the absence of any additional input the neuron will remain at its current state. 

Other notes: If $T_i < 0$, the neuron will spike with probability $p_i$ on each simulation step, assuming that the input is not sufficiently negative to pull the neuron's internal state below threshold.  If $p_i = 1.0$, then the neuron model is deterministic.

#### Input Neurons
Thus far in Fugu development, we have used two types of neurons to provide input to the network.  In the first case, spike times of the input neurons are pre-determined prior to run-time, as in `Example_Brick_Building.ipynb.`   

In the second case, input neurons are constructed to spike as a function of input data that is provided at run-time.  In this scenario, these input neurons act to "translate" information from the data domain to the spiking domain.  


### Synapses
Thus far in Fugu, all neurons are modeled as point neurons with at most one synapse between a given pair pre- and post-synaptic neurons.  

In addition to being characterized by a weight (see below), each synapse is also characterized by a delay (time of transmission between pre-synaptic spike to post-synaptic potential).


| Record Key | Definition | Type |
| ----- | ----- | ----- |
| 'weight' | Synaptic Weight | float | 
| 'delay' | Synaptic Delay | int [1, inf) |


Currently Fugu only supports discrete-time simulation.  Therefore delays must take integer values of at least 1 simulation step.  Synaptic weights may take any floating point value, including negative (inhibitory) values.










# I THINK THE REST IS OUT OF SCOPE?  (EITHER NOT FUGU OR NOT NEURON MODEL)
## Basic Operation

### Recording from Neurons
As referenced above, the 'record' neuron property defines which quantities should be recorded 
during simulation at each timestep.  Recordings are stored within the corresponding 
neuron tensor dictionary (see Basic Operation below).  The node property 
'record' should take the value of a list with any of the following values:

| Record Key | Definition | Type |
| ---- | ---- | ---- |
| 'spikes' | Spike raster | torch.Tensor, byteArray |
| 'potential' | Potential after the neuron is updated | torch.Tensor, float|
| 'preactivation' | Potential before the threhold | torch.Tenros, float|

### Defining a Network Graph
Currently there are two methods for defining a network graph:

1. Define a NetworkX DiGraph according to the Network Specification section. This 
method is relatively straightforward due to the easy syntax of networkx.  Portions of 
the network can be defined using python control statements or list comprehensions.  
For example, `[graph.add_edge(0,i, weight=1.0, delay=1) for i in range(1,10)]` will 
add a synapse with weight 1.0, delay 1 between neuron 0 and each of the neurons 1 through 9.
This allows for quick definitions of randomly distributed connections via numpy.random.  
Graph properties can be accessed using `graph.graph['property']`, node properties can be 
accessed using `graph.nodes[node]['property']`, and edge properties can be accessed by 
`graph[from_node][to_node]['property']`.  Though it is not necessary, it is suggested to use 
integers for node names.
2. Convert a SpikingPDE network.  The function `ds.SpikingPDE_to_ds_graph(graph, neuron_list)` 
requires two objects, the SpikingPDE network graph and the SpikingPDE neuron_list.  
The function returns a complete ds-compatible network graph object.  This function 
copies the original graph, so large graphs may require a large amount of memory.

```python
# A Simple Conversion Example using SpikingPDE.MarkovNetwork

#Defining the node transitions
transitions = {}
for i in range(0,19):
    transitions[(i,)] = SpikingPDE.Transition(location=(i,), neighbors=[((i+1,),0.5)])
transitions[(19,)] = SpikingPDE.Transition(location=(9,),neighbors=[])
for i in range(0,19):
    transitions[(i+1,)].neighbors.append(((i,), 0.5))
transitions[(0,)].neighbors.append(((19,),0.5))
transitions[(19,)].neighbors.append(((0,),0.5))

#Creating the network
net = SpikingPDE.MarkovNetwork(initial_walkers={(10,):30, (13,):30},
                    transitions=transitions,
                    synchronized=True,
                    log_potential=True, log_spikes=True)
graph = net.build()

#The object graph now contains connectivity information
#Neuron information is contained in SpikingPDE.neuron_list
graph = SpikingPDE_to_ds_graph(graph, SpikingPDE.neuron_list)

#The object graph is ds-compatible and contains information on both
#the connectivity and the neurons

#The below code will allow us to record stats from 'counter' neurons
recorded_neuron_groups = ['counter']
for node in graph.nodes():
    for group in graph.nodes[node]['groups']:
        if group in recorded_neuron_groups:
            graph.nodes[node]['record'] = ['spikes', 'preactivation', 'potential']
            
#We can then convert initial walkers to an appropriate torch.Tensor
injection = {0:get_injection_tensor_from_initial_walkers({(10,):30, (13,):30},
                                                        graph, 
                                                        graph.number_of_nodes())}

#Additionally, we need to start the network by having a supervisor neuron spike
injection[0][net.walker_supervisor.name] = 10

#Lastly, we run a simulation for 1000 timesteps
#result will contain a dictionary of neuron batches, each with tensors 
#detailing neuron states
result = run_simulation(graph, 1000, injection, batch_size=100, verbose=1)
```

### Preparing Current Injection
To provide external input to the network, you can specify current injection values. 
The current injection method is flexible: you can use it to add a specific value 
to the potential of a neuron, you can mimic input spikes, or you can drive designated 
input neurons for broadcast spiking behavior.  You should prepare a python 
dictionary indexed by the timestep of the injection values.  Injection values 
should be a 1-D tensor (float) of length equal to the number of neurons.

### Running a network
To run a network, simply call 
`ds.run_simulation(graph, timesteps, injection_dictionary, start_time=0, 
batch_size=1000, watches=[], verbose=10)`
where:
- `graph` network graph object
- `timesteps` length of the simulation (integer)
- `injection_dictionary`  a (possibly empty) dictionary of network injection values
- `start_time` optional integer value for the starting time of the simulation (integer)
- `batch_size` maximum size of the neuron tensors for computation (integer)
- `watches` list of simulation 'watches' (described below)
- `verbose` 0 for terse output, 1 for verbose output

The function will return a 'neuron tensors' dictionary.  Within the dictionary will 
be groups of neuron states. Each neuron group will of size `batch_size` or smaller, 
and neurons will be grouped by inferred neuron type.  Indexes are arbitrary. 
Recorded values will be stored here in `spike_history`, `potential_history` and 
`preactivation_history` tensors.  Tensors can be converted to lists using `Tensor.tolist()`.

### Creating Simulation Watches
The simulator supports a watch or callback mechanism.  Recording spikes, potentials 
and preactivations are built into the appropriate neuron types (again, automatically 
inferred by the neuron properties).  However, more advanced recording (e.g. updating plots, 
learning rules, population statistics) require the use of a watch object.  Watch objects do 
not need to inherit from a particular base class (yet).  Instead, the only requirement is that they 
implement a callback-style function.  Right now, there is only one, but we can add more easily.

| Function | Time-of-Call |
| ---- | ---- |
| `on_end_timestep(self, neuron_tensors, new_spikes)` | At the end of a timestep after all neuron dynamics have been calculated |


## Examples
There are two basic examples available in test_networks.py.  This script creates two 
simple test networks.  The first is a small network used to test basic neuron and 
synapse properties.  The second is a basic 1-D random walk density method.

Additionally, there is hebbian_example.py  which shows a very simple network 
that supports hebbian learning using the watch mechanism. 

More examples to follow soon.