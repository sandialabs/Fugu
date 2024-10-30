## Fugu Neural Networks

### Specifying the network in Fugu
To prepare a network, fugu "bricks" are assembled into a "scaffold".  Each brick may be thought of as generating a graph component analagous to a local neural "circuit" with neurons and synapses (as defined by nodes and edges within the brick).  The scaffold then defines the connectivity between these local neural circuits.  As with any neural circuit or graph, the functional properties of a brick's circuitry are determined by the specific properties of the neurons and synapses as well as the circuit connectivity.  Below we describe the model properties of the neurons and synapses currently available within Fugu.  

It should be noted that each brick is the algorithm for generating a neural circuit, not the graph itself, and that the network object is not constructed until `lay_bricks()` is called.  At evaluation this network is converted into torch.Tensor objects for simulation.  


### Neuron Model
The Fugu neuron model is based upon a standard leaky-integrate-and-fire neuron.  The fugu neuron is constructed with the variables (for neuron $i$):

Variable Key | Definition/Description | Variable Name | Type | 
| ----- | ----- | ----- | ---- |
| potential | Internal state of the neuron, analagous to the biological membrane potential | $`x_i`$ | float |
| threshold | Neuron spike threshold | $`T_i`$ | float |
| decay | Decay constant | $`m_i`$ | float $`\in [0,1]`$|
| p | Probability of spike (given that $`x_i > T_i`$) | $`p_i`$ | float $`\in [0,1]`$| |
| record | List of values to record | N/A | list of strings |
| N/A | Random Draw | a | float $`\in [0,1]`$ |


For each simulation step $`t`$, $`x_i`$ is computed as follows: 

1. $`x_i = x_{t-1,i} + I_i + W_{i}*S_{t-1}`$,  
where $`I_i`$ is the external current injection to neuron $`i`$, $`W_{i}`$ is a vector containing the weights of the edges between every other neuron in the network and neuron $`i`$, and $`S_{t-1}`$ is a vector containing the spike history of the network for the previous stimulation step.   

2. If $`x_i > T_i`$, then if $`a < p_i`$, then $`S[i]`$  
If the internal state of neuron $i$ is greater than its spike threshold, spike with probability $`p_i`$.  Currently the internal state is reset to $`0`$ following a spike.  

3. If $`x_i <= T_i`$, then $`x_i = x_i * (1-m_i)`$ 
If neuron $`i`$ does not spike, the internal state decays to $`0`$ as described above.  Note: if $`m_i = 0`$ there is no decay and in the absence of any additional input the neuron will remain at its current state. 

Other notes: If $`T_i < 0`$, the neuron will spike with probability $`p_i`$ on each simulation step, assuming that the input is not sufficiently negative to pull the neuron's internal state below threshold.  If $`p_i = 1.0`$, then the neuron model is deterministic.

#### Input Neurons
Thus far in Fugu development, we have used two types of neurons to provide input to the network.  In the first case, spike times of the input neurons are pre-determined prior to run-time, as in `Example_Brick_Building.ipynb.`   

In the second case, input neurons are constructed to spike as a function of input data that is provided at run-time.  In this scenario, these input neurons act to "translate" information from the data domain to the spiking domain.  


### Synapses
Thus far in Fugu, all neurons are modeled as point neurons with at most one synapse between a given pair pre- and post-synaptic neurons.  

In addition to being characterized by a weight (see below), each synapse is also characterized by a delay (time of transmission between pre-synaptic spike to post-synaptic potential).


| Record Key | Definition | Type |
| ----- | ----- | ----- |
| 'weight' | Synaptic Weight | float | 
| 'delay' | Synaptic Delay | int $`[1, \inf)`$ |


Currently Fugu only supports discrete-time simulation.  Therefore delays must take integer values of at least 1 simulation step.  Synaptic weights may take any floating point value, including negative (inhibitory) values.



