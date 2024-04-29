# isort: skip_file
# fmt: off
import numpy as np
from fugu.bricks import Vector_Input
from .bricks import Brick

from collections import deque

class Mock_Input(Vector_Input):
    def __init__(self,spikes,metadata,time_dimension=False,coding='Undefined',batchable=True,name="MockInput"):
        super(Mock_Input, self).__init__(spikes,time_dimension,coding,batchable,name)
        if type(metadata) is list:
            self.metadata = {**metadata[0], **self.metadata}
        else:
            self.metadata = {**metadata, **self.metadata}

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):

        if type(metadata) is list:
            self.metadata = {**metadata[0], **self.metadata}
        else:
            self.metadata = {**metadata, **self.metadata}

        if not self.time_dimension:
            self.vector = np.expand_dims(self.vector, len(self.vector.shape))

        complete_node = self.generate_neuron_name("complete")
        begin_node = self.generate_neuron_name("begin")
        vector_size = len(self.vector) * len(self.vector.shape)

        time_length = self.vector.shape[-1]
        if time_length == 1:
            graph.add_node(complete_node,index=-1,threshold=0.0,decay=0.0,p=1.0,potential=0.1)
        else:
            graph.add_node(complete_node,index=-1,threshold=0.5,decay=0.0,p=1.0,potential=0.0)
            graph.add_edge(begin_node,complete_node,weight=1.0,delay=time_length - 1)
            
        output_lists = [[]]
        output_codings = [self.coding]
        self.is_built = True

        return (graph, self.metadata, [{"complete": complete_node, "begin": begin_node}], output_lists, output_codings,)


class Mock_Brick(Brick):
    def __init__(self,fugu_brick,metadata,name="MockBrick"):
        super(Mock_Brick, self).__init__(name)
        for key, value in vars(fugu_brick).items():
            setattr(self,key,value)
        if type(metadata) is list:
            self.metadata = {**metadata[0], **fugu_brick.metadata}
        else:
            self.metadata = {**metadata, **fugu_brick.metadata}

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):

        # if type(metadata) is list:
        #     self.metadata = {**metadata[0], **self.metadata}
        # else:
        #     self.metadata = {**metadata, **self.metadata}

        if not self.time_dimension:
            self.dvector = np.expand_dims(self.vector, len(self.vector.shape))

        complete_node = self.generate_neuron_name("complete")
        begin_node = self.generate_neuron_name("begin")
        vector_size = len(self.vector) * len(self.vector.shape)

        time_length = self.dvector.shape[-1]
        if time_length == 1:
            graph.add_node(complete_node,index=-1,threshold=0.0,decay=0.0,p=1.0,potential=0.1)
        else:
            graph.add_node(complete_node,index=-1,threshold=0.5,decay=0.0,p=1.0,potential=0.0)
            graph.add_edge(begin_node,complete_node,weight=1.0,delay=time_length - 1)
            
        output_lists = [[]]
        self.index_map = np.ndindex(self.vector.shape[:-1])

        input_shape = self.vector.shape
        # for i, index in enumerate(self.index_map):
        for row in np.arange(0,input_shape[1]):
            for col in np.arange(0,input_shape[2]):
                for channel in np.arange(0,input_shape[3]):
                    neuron_name = self.generate_neuron_name(f"{channel}{row}{col}")

                    graph.add_node(neuron_name,
                                index=(row,col,channel),
                                threshold=0.0,
                                decay=0.0,
                                p=1.0)
                    output_lists[0].append(neuron_name)
        output_codings = [self.coding]
        self.is_built = True

        return (graph, self.metadata, [{"complete": complete_node, "begin": begin_node}], output_lists, output_codings,)

    def __iter__(self):
        self.current_time = 0
        return self

    def __next__(self):
        if self.vector.shape[-1] > self.current_time:
            self.current_time += 1
            this_time_vector = self.vector[..., self.current_time - 1]
            local_idxs = np.array(np.where(this_time_vector))
            num_spikes = len(local_idxs[0])
            global_idxs = deque()
            for spike in range(num_spikes):
                idx_to_build = deque()
                for dimension in range(len(local_idxs)):
                    idx_to_build.append(local_idxs[dimension][spike])
                global_idxs.append(tuple(idx_to_build))
            spiking_neurons = [
                self.generate_neuron_name(str(idx)) for idx in global_idxs
            ]
            return spiking_neurons
        else:
            raise StopIteration