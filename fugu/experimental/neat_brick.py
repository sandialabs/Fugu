#!/usr/bin/env python
# coding: utf-8

# This contains a brick designed to be used with neat-python
# The package neat-python uses configuration files to define an evolutionary learning scheme
# It implements the NEAT approach in pure python
# neat-python : https://github.com/CodeReclaimers/neat-python
# NEAT: Stanley, Kenneth O., and Risto Miikkulainen. 
#       "Evolving neural networks through augmenting topologies." 
#       Evolutionary computation 10.2 (2002): 99-127.
#
# Addtional dependencies: neat-python
#
# Currently only the default feed-forward genome is supported
#
# Multiple inputs will be concatenated into a single dimension
#
# Interface, design and output may change in future versions
#

from fugu.bricks import Brick
import neat
from neat.graphs import feed_forward_layers
import numpy as np 
import networkx as nx


class NEAT_FF(Brick):
    def __init__(self,
                 genomes,
                 genome_config,
                 buffer=False,
                 name='NEAT'):
        super(NEAT_FF, self).__init__(name)
        self.genomes = genomes  #neat-python genome tuples
        self.genome_config = genome_config  #neat-python config
        self.name = name
        self.buffer = buffer  #Copy input neurons within each genome
        self.is_built = False
        self.genome_delays = dict()
    def update_genomes(self, genomes):
        #For when genomes need to be redefined, e.g. each generation
        #You still need to lay_bricks (set attributes may be an alternative in next verison)
        self.genomes = genomes  
    def get_genome_neuron_number(self, genome_name, neuron_number):
        genome_prefix = "" if genome_name is None else "{}_".format(genome_name)
        neuron_name = "{}{}".format(genome_prefix, neuron_number)
        neuron_name = self.generate_neuron_name(neuron_name)
        return neuron_name
    def _instantiate_genome(self, genome, graph, input_list, control_nodes, genome_name=None):
        input_list = np.asarray(input_list)  #Always want 1 dimensional
        input_list = input_list.flatten()
        
        if self.buffer:
            new_input_list = []
            for idx, input_neuron in enumerate(input_list):
                name = self.get_genome_neuron_number(genome_name, 'buffer_{}'.format(input_neuron))
                graph.add_node(name,
                               threshold=0.5,
                               decay = 1.0,
                               p=1.0)
                graph.add_edge(input_neuron,
                               name,
                               weight=1.0,
                               delay=1)
                new_input_list.append(name)
            input_list = new_input_list
        
        output_neurons = []
        this_genome_neurons = []
        
        def neuron_name_from_number(neuron_no):
            if neuron_no >= 0:
                neuron_name = self.get_genome_neuron_number(genome_name, neuron_no)
                return neuron_name
            else:
                return input_list[-neuron_no-1] #Input neurons are indexed -1, -2, ...
        layers = feed_forward_layers(self.genome_config.input_keys, self.genome_config.output_keys, [c for c in genome.connections if genome.connections[c].enabled]  ) 
        num_layers = max(len(layers),1)
        
        if self.buffer:
            num_layers +=1
        
        self.genome_delays[genome_name] = num_layers
        if num_layers == 0:
            print(genome)
            print(layers)
            print([c for c in genome.connections if genome.connections[c].enabled] )
            print(genome_name)
            exit()
        neuron_layers = dict()
        for layer_no, layer in enumerate(layers):
            for neuron in layer:
                neuron_layers[neuron] = layer_no + 1
        for key in self.genome_config.input_keys:
            neuron_layers[key] = 0

        for neuron_no in genome.nodes:
            neuron = genome.nodes[neuron_no]
            threshold = -neuron.bias 
            decay = 1.0
            neuron_name = neuron_name_from_number(neuron_no)
            graph.add_node(neuron_name,
                           threhold=threshold,
                           decay=decay,
                           index=(neuron_no,),
                           p=1.0,
                           )
            if neuron_no in self.genome_config.output_keys:
                output_neurons.append(neuron_name)
            this_genome_neurons.append(neuron_name)
            
        for source, target in genome.connections:
            conn = genome.connections[(source, target)]
            if not conn.enabled:
                continue
            weight = conn.weight 
            if source in neuron_layers and target in neuron_layers:
                delay = neuron_layers[target]-neuron_layers[source]
            else:
                delay = 1 
            source_name = neuron_name_from_number(source)
            target_name = neuron_name_from_number(target)
            graph.add_edge(source_name, target_name, weight=weight, delay=delay)

        new_complete_node_name = self.generate_neuron_name('complete')
        graph.add_node(new_complete_node_name,
                index=-1,
                threshold=0.9,
                decay=0.0,
                p=1.0,
                potential=0.0,
                )
        graph.add_edge(control_nodes[0]['complete'], new_complete_node_name, weight=1.5, delay = num_layers)
        return output_neurons, {'complete':new_complete_node_name}
        
    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        if len(input_lists) == 0:
            raise ValueError("Need at least one input")
        combined_input = None
        for idx, il in enumerate(input_lists):
            if idx == 0:
                combined_input = [n for n in il] #Just a copy of the first input
            else:
                combined_input.extend(il) #Add the neurons from any other input list
                        
        output_neurons = []
        new_control_nodes = []
        for genome_num, genome in self.genomes:
            on, cn = self._instantiate_genome(genome, graph, combined_input, control_nodes, genome_name=genome_num)
            output_neurons.append(on)
            new_control_nodes.append(cn)
        self.is_built = True
        self.metadata = {'num_genomes': len(self.genomes),
                         }
        return (graph,
               self.metadata,
                control_nodes,
                output_neurons,
                ['raster']
               )
