import numpy as np

from .bricks import Brick
from ..scaffold import ChannelSpec, PortData, PortSpec, PortUtil


class NoOps(Brick):
    """
    Brick that performs no operation (i.e., does nothing).
    All codings are supported.
    """

    def __init__(self, name="NoOps", noops_delay: int = 10):
        """
        Args:
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(NoOps, self).__init__(name)

        # The brick hasn't been built yet.
        self.is_built = False

        # We just store the name passed at construction.
        self.name = name
        self.noops_delay = noops_delay

    # This method describes the input ports that any AND brick expects.
    # This includes the actual data values, as well as a signal indicating when the
    # source of input is ready for us to procss the data. Since we execute in a single
    # cycle, this signal is simply passed on to the next brick (see build2() later).
    @classmethod
    def input_ports(cls) -> dict[str, PortSpec]:
        port = PortSpec(name="input", minimum=0, maximum=0)
        port.channels["data"] = ChannelSpec(name="data")
        port.channels["complete"] = ChannelSpec(name="complete")
        return {port.name: port}

    # This method describes the output port that any AND brick provides.
    # This includes the data and the 'complete' signal that we pass on.
    @classmethod
    def output_ports(cls) -> dict[str, PortSpec]:
        port = PortSpec(name="output", minimum=0, maximum=0)
        port.channels["data"] = ChannelSpec(name="data")
        port.channels["complete"] = ChannelSpec(name="complete")
        return {port.name: port}

    # This method does the actual network construction. It reads the input ports, wires up
    # some neurons, and provides their identities via the output port.
    def build2(self, graph, inputs: dict[str, PortData] = {}):
        """
        Build NoOps brick.
        """

        # Set up convenience variables for accessing our working ports.
        input_tuple = PortUtil.get_autoports(inputs, "input", 0)  # Unpack the input ports.
        result = PortUtil.make_ports_from_specs(NoOps.output_ports())  # Create our output port(s).
        output = result["output"]  # Unpack the only actual output port.
        data = output.channels["data"]  # Unpack the data channel where our main result goes.

        # Hook up the signals.
        # We just forward the incoming signal
        complete_node_name = self.generate_neuron_name("complete")
        graph.add_node(complete_node_name, index=-1, threshold=0.0, decay=0.0, p=1.0, potential=0.0)
        for input_id in input_tuple:
            graph.add_edge(input_id.channels["complete"].neurons[0], complete_node_name, weight=1.0, delay=self.noops_delay)

        # Build the computational graph.
        # The plan is to pass the input data downstream to the output node.
        for input_id in input_tuple:
            neurons = input_id.channels["data"].neurons
            for neuron in neurons:
                output_node = self.generate_neuron_name(f"output_{neuron}")
                data.neurons.append(output_node)

                graph.add_node(output_node, index=0, threshold=1.0, decay=1.0, p=1.0, potential=0.0)
                graph.add_edge(neuron, output_node, weight=1.0, delay=self.noops_delay)

        self.is_built = True
        return result
