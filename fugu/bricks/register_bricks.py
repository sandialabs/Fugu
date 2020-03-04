#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .bricks import Brick, input_coding_types


class Register(Brick):
    '''
    Brick that stores the binary encoding of an non-negative integer.
    '''
    def __init__(self, max_size, initial_value=0, name=None, output_coding='Undefined'):
        super(Register, self).__init__("Register")
        self.is_built = False
        self.name = name
        self.supported_codings = input_coding_types

        self.output_codings = [output_coding]
        self.metadata = {'D': None}

        self.max_size = max_size
        self.initial_value = initial_value

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Register brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All coding types supported

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError(
                        "Unsupported Input Coding. Found: {}. Allowed: {}".format(
                                                                             input_coding,
                                                                             self.supported_codings,
                                                                             )
                        )
        begin_node_name = self.generate_neuron_name('begin')
        graph.add_node(begin_node_name, threshold=0.1, decay=0.0, potential=0.0)

        complete_name = self.generate_neuron_name('complete')
        graph.add_node(complete_name, threshold=0.1, decay=0.0, potential=0.0)
        complete_node_list = [complete_name]

        input_value_bits = input_lists[0]
        recall_input = input_lists[1][-1]
        clear_input = input_lists[2][-1]
        set_input = input_lists[3][-1]

        register_name_base = "slot_{}"
        output_name_base = "output_{}"
        recall_control_name = "recall_control"
        recall_name_base = "recall_bit{}"
        clear_control_name = "clear_control"
        clear_name_base = "clear_bit{}"
        set_control_name = "set_control"
        set_name_base = "set_bit{}"

        graph.add_node(
                recall_control_name,
                threshold=0.9,
                decay=0.0,
                potential=0.0,
                )
        graph.add_node(
                set_control_name,
                threshold=0.9,
                decay=0.0,
                potential=0.0,
                )
        graph.add_node(
                clear_control_name,
                threshold=0.9,
                decay=0.0,
                potential=0.0,
                )

        graph.add_edge(
                recall_input,
                recall_control_name,
                weight=1.0,
                delay=1.0,
                )
        graph.add_edge(
                recall_input,
                begin_node_name,
                weight=1.0,
                delay=1.0,
                )
        graph.add_edge(
                recall_input,
                complete_name,
                weight=1.0,
                delay=4.0,
                )
        graph.add_edge(
                clear_input,
                clear_control_name,
                weight=1.0,
                delay=1.0,
                )
        graph.add_edge(
                clear_input,
                begin_node_name,
                weight=1.0,
                delay=1.0,
                )
        graph.add_edge(
                clear_input,
                complete_name,
                weight=1.0,
                delay=3.0,
                )
        graph.add_edge(
                set_input,
                set_control_name,
                weight=1.0,
                delay=1.0,
                )
        graph.add_edge(
                set_input,
                begin_node_name,
                weight=1.0,
                delay=1.0,
                )
        graph.add_edge(
                set_input,
                complete_name,
                weight=1.0,
                delay=5.0,
                )

        outputs = []

        # determine initial states
        initial_value = self.initial_value
        bit_string = [0.0 for i in range(self.max_size)]
        for i in range(self.max_size - 1, -1, -1):
            power_of_2 = 2 ** i
            if power_of_2 <= initial_value:
                initial_value -= power_of_2
                bit_string[i] = 1.0

        for i in range(self.max_size):
            register_name = self.generate_neuron_name(register_name_base.format(i))
            output_name = self.generate_neuron_name(output_name_base.format(i))
            recall_name = self.generate_neuron_name(recall_name_base.format(i))
            clear_name = self.generate_neuron_name(clear_name_base.format(i))
            set_name = self.generate_neuron_name(set_name_base.format(i))
            outputs.append(output_name)

            # Create register slot
            graph.add_node(
                    register_name,
                    threshold=1.99,
                    decay=0.0,
                    potential=bit_string[i],
                    )
            graph.add_node(
                    output_name,
                    threshold=1.99,
                    decay=1.0,
                    potential=0.0,
                    bit_position=i,
                    )
            graph.add_node(
                    recall_name,
                    threshold=0.99,
                    decay=0.0,
                    potential=0.0,
                    )
            graph.add_node(
                    clear_name,
                    threshold=0.99,
                    decay=1.0,
                    potential=0.0,
                    )
            graph.add_node(
                    set_name,
                    threshold=1.99,
                    decay=1.0,
                    potential=0.0,
                    )

            # Connect controls to slot
            if i < len(input_value_bits):
                graph.add_edge(
                        input_value_bits[i],
                        set_name,
                        weight=1.0,
                        delay=2.0,
                        )
            graph.add_edge(
                    set_control_name,
                    clear_control_name,
                    weight=1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    set_control_name,
                    set_name,
                    weight=1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    set_name,
                    register_name,
                    weight=1.0,
                    delay=3.0,
                    )
            graph.add_edge(
                    recall_control_name,
                    recall_name,
                    weight=1.0,
                    delay=2.0,
                    )
            graph.add_edge(
                    recall_control_name,
                    register_name,
                    weight=1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    recall_control_name,
                    output_name,
                    weight=1.0,
                    delay=2.0,
                    )
            graph.add_edge(
                    clear_name,
                    register_name,
                    weight=-1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    clear_control_name,
                    clear_name,
                    weight=1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    clear_control_name,
                    register_name,
                    weight=1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    register_name,
                    register_name,
                    weight=1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    register_name,
                    recall_name,
                    weight=-1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    register_name,
                    output_name,
                    weight=1.0,
                    delay=1.0,
                    )
        self.is_built = True

        output_lists = [outputs]

        return (
                 graph,
                 self.metadata,
                 [{'complete': complete_node_list[0], 'begin': begin_node_name}],
                 output_lists,
                 self.output_codings,
                 )


class Max(Brick):
    '''
    Brick that calculates the maximum value of a collection of values stored as binary registers.
    '''

    def __init__(self, name=None, output_coding='Undefined'):
        super(Max, self).__init__("Max")
        self.name = name
        self.supported_codings = input_coding_types

        self.output_codings = [output_coding]
        self.metadata = {'D': None}

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Register brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All coding types supported

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError(
                        "Unsupported Input Coding. Found: {}. Allowed: {}".format(
                                                                             input_coding,
                                                                             self.supported_codings,
                                                                             )
                        )

        begin_node_name = self.generate_neuron_name('begin')
        graph.add_node(
                begin_node_name,
                threshold=0.1,
                decay=0.0,
                potential=0.0,
                )

        complete_name = self.generate_neuron_name('complete')
        graph.add_node(
                complete_name,
                threshold=0.1,
                decay=0.0,
                potential=0.0,
                )
        complete_node_list = [complete_name]

        max_size = 0
        for register in input_lists:
            register_size = len(register)
            if register_size > max_size:
                max_size = register_size
            for bit in register:
                graph.add_edge(
                        bit,
                        begin_node_name,
                        weight=1.0,
                        delay=1.0,
                        )

        max_time = 3 + 4 * max_size
        graph.add_edge(
                begin_node_name,
                complete_name,
                weight=1.0,
                delay=max_time,
                )

        m_base = "M_{}"
        copy_base = "c_{}_{}"
        or_base = "OR_{}"
        valid_base = "V_{}_{}"
        active_base = "a_{}_{}"
        intercept_base = "I_{}_{}"

        m_names = []

        for j in range(max_size):
            # M_j
            m_j = self.generate_neuron_name(m_base.format(j))
            graph.add_node(
                    m_j,
                    threshold=0.5,
                    decay=0.0,
                    potential=0.0,
                    )
            m_names.append(m_j)

            # OR_j
            graph.add_node(
                    self.generate_neuron_name(or_base.format(j)),
                    threshold=0.5,
                    decay=0.0,
                    potential=0.0,
                    )

        for i, register_bits in enumerate(input_lists):
            intercept_index = max_size - 1
            # Setup Layer

            a_i_I = self.generate_neuron_name(active_base.format(i, 'I'))
            # a_i_I
            graph.add_node(
                    a_i_I,
                    threshold=0.5,
                    decay=1.0,
                    potential=0.0,
                    )

            for bit in register_bits:
                graph.add_edge(
                        bit,
                        a_i_I,
                        weight=1.0,
                        delay=1.0,
                        )

            # First Layer
            # a_i_L
            a_i_L = self.generate_neuron_name(active_base.format(i, intercept_index))
            graph.add_node(
                    a_i_L,
                    threshold=0.5,
                    decay=1.0,
                    potential=0.0,
                    )
            # I_i_L
            I_i_L = self.generate_neuron_name(intercept_base.format(i, intercept_index))
            graph.add_node(
                    I_i_L,
                    threshold=0.5,
                    decay=0.0,
                    potential=0.0,
                    )

            graph.add_edge(
                    a_i_I,
                    a_i_L,
                    weight=1.0,
                    delay=2.0,
                    )
            graph.add_edge(
                    register_bits[intercept_index],
                    I_i_L,
                    weight=-1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    register_bits[intercept_index],
                    self.generate_neuron_name(or_base.format(intercept_index)),
                    weight=1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    self.generate_neuron_name(or_base.format(intercept_index)),
                    I_i_L,
                    weight=1.0,
                    delay=1.0,
                    )
            graph.add_edge(
                    I_i_L,
                    a_i_L,
                    weight=-1.0,
                    delay=1.0
                    )

            # Middle Layers
            for j in range(2, max_size + 1):
                intercept_index = max_size - j

                prev_active_name = self.generate_neuron_name(active_base.format(i, intercept_index + 1))
                curr_active_name = self.generate_neuron_name(active_base.format(i, intercept_index))
                intercept_name = self.generate_neuron_name(intercept_base.format(i, intercept_index))
                valid_name = self.generate_neuron_name(valid_base.format(i, intercept_index))
                or_name = self.generate_neuron_name(or_base.format(intercept_index))

                # a_i_j
                graph.add_node(
                        curr_active_name,
                        threshold=0.5,
                        decay=1.0,
                        potential=0.0,
                        )
                # I_i_j
                graph.add_node(
                        intercept_name,
                        threshold=0.5,
                        decay=0.0,
                        potential=0.0,
                        )
                # V_i_j
                graph.add_node(
                        valid_name,
                        threshold=1.9,
                        decay=1.0,
                        potential=0.0,
                        )

                graph.add_edge(
                        prev_active_name,
                        curr_active_name,
                        weight=1.0,
                        delay=4.0,
                        )
                graph.add_edge(
                        prev_active_name,
                        valid_name,
                        weight=1.0,
                        delay=1.0,
                        )
                graph.add_edge(
                        register_bits[intercept_index],
                        valid_name,
                        weight=1.0,
                        delay=4 * (j - 1),
                        )
                graph.add_edge(
                        valid_name,
                        intercept_name,
                        weight=-1.0,
                        delay=1.0,
                        )
                graph.add_edge(
                        valid_name,
                        or_name,
                        weight=1.0,
                        delay=1.0,
                        )
                graph.add_edge(
                        or_name,
                        intercept_name,
                        weight=1.0,
                        delay=1.0,
                        )
                graph.add_edge(
                        intercept_name,
                        curr_active_name,
                        weight=-1.0,
                        delay=1.0,
                        )

            # Copy and Output Layer
            for j in range(max_size):
                copy_name = self.generate_neuron_name(copy_base.format(i, j))
                # c_i_j
                graph.add_node(
                        copy_name,
                        threshold=1.9,
                        decay=1.0,
                        potential=0.0,
                        )

                graph.add_edge(
                        curr_active_name,
                        copy_name,
                        weight=1.0,
                        delay=1.0,
                        )
                graph.add_edge(
                        register_bits[j],
                        copy_name,
                        weight=1.0,
                        delay=4 * (max_size),
                        )
                graph.add_edge(
                        copy_name,
                        self.generate_neuron_name(m_base.format(j)),
                        weight=1.0,
                        delay=1.0,
                        )

        output_lists = [m_names]

        self.is_built = True

        return (
                 graph,
                 self.metadata,
                 [{'complete': complete_node_list[0], 'begin': begin_node_name}],
                 output_lists,
                 self.output_codings,
                 )


class Adder(Brick):
    '''
    Brick that calculates the maximum value of a collection of values stored as binary registers.
    '''

    def __init__(self, name=None, output_coding='Undefined'):
        super(Adder, self).__init__("Adder")
        self.name = name
        self.supported_codings = input_coding_types

        self.output_codings = [output_coding]
        self.metadata = {'D': None}

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Register brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All coding types supported

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """
        if len(input_lists) > 2:
            raise ValueError(
                    "Too many inputs! {} can only support two inputs, received: {}".format(
                                                                                      self.brick_tag,
                                                                                      len(input_lists),
                                                                                      )
                    )
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError(
                        "Unsupported Input Coding. Found: {}. Allowed: {}".format(
                                                                             input_coding,
                                                                             self.supported_codings,
                                                                             )
                        )

        begin_node_name = self.generate_neuron_name('begin')
        graph.add_node(
                begin_node_name,
                threshold=0.1,
                decay=0.0,
                potential=0.0,
                )

        complete_name = self.generate_neuron_name('complete')
        graph.add_node(
                complete_name,
                threshold=0.1,
                decay=0.0,
                potential=0.0,
                )
        complete_node_list = [complete_name]

        max_size = 0
        for register in input_lists:
            register_size = len(register)
            if register_size > max_size:
                max_size = register_size
            for bit in register:
                graph.add_edge(
                        bit,
                        begin_node_name,
                        weight=1.0,
                        delay=1.0,
                        )

        graph.add_edge(
                begin_node_name,
                complete_name,
                weight=1.0,
                delay=1,
                )

        carry_base = "C_{}"
        S_base = "S_{}"

        S_names = []
        for i in range(max_size + 1):
            carry_name = self.generate_neuron_name(carry_base.format(i))
            S_name = self.generate_neuron_name(S_base.format(i))
            S_names.append(S_name)

            graph.add_node(
                    carry_name,
                    threshold=2 ** (i + 1) - 0.01,
                    decay=1.0,
                    potential=0.0,
                    )
            graph.add_node(
                    S_name,
                    threshold=0.99,
                    decay=1.0,
                    potential=0.0,
                    bit_position=i,
                    )

            graph.add_edge(
                    carry_name,
                    S_name,
                    weight=-2.0,
                    delay=1.0,
                    )
            if i > 0:
                prev_carry_name = self.generate_neuron_name(carry_base.format(i - 1))

                graph.add_edge(
                        prev_carry_name,
                        S_name,
                        weight=1.0,
                        delay=1.0,
                        )

        for register in input_lists:
            for i, bit in enumerate(register):
                S_name = self.generate_neuron_name(S_base.format(i))
                graph.add_edge(
                        bit,
                        S_name,
                        weight=1.0,
                        delay=2.0,
                        )

                for j in range(i, max_size):
                    carry_name = self.generate_neuron_name(carry_base.format(j))
                    graph.add_edge(
                            bit,
                            carry_name,
                            weight=2 ** i,
                            delay=1.0,
                            )

        output_lists = [S_names]

        self.is_built = True

        return (
                 graph,
                 self.metadata,
                 [{'complete': complete_node_list[0], 'begin': begin_node_name}],
                 output_lists,
                 self.output_codings,
                 )
