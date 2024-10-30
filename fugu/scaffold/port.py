#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field

@dataclass
class ChannelSpec:
    """
        Describes the characteristics of one group of neurons.
        These work together for a purpose designated by the brick.
        A Port can have several different Channels. Typically, these
        would include 'data' and a control signal such as 'complete' or 'begin'.
    """
    name:        str
    description: str        = ''                          # Human-readable documentation about Channel.
    coding:      list[str]  = field(default_factory=list) # For input, all compatible codings. For output, generally just the one expected coding, but can be several if all compatible.
    shape:       tuple[int] = None                        # The tensor arrangement of neurons. None indicates no particular expectation. If any element in the list is <=0, it indicates unknown size for that dimension.
    required:    bool       = True                        # For input ports, indicates that this channel must be supplied by the source port for the brick to work correctly. Ignored for output ports.

@dataclass
class PortSpec:
    """
        Describes the characteristics of one input or output connection of a brick.
        A brick may have several of each kind.
        Each port may have several channels in it, each conveying a different kind of
        information associated with the port.

        In some cases, a brick may receive a variable number of inputs. To handle this,
        a brick may declare an "auto-port" by setting maximum to something other than 1.
    """
    name:        str                                                  # Concise name of port, but generally human readable. The names '0', '1', '2', and so on, are reserved to mean the port with index==int(name).
    description: str                    = ''                          # Human-readable documentation about Port.
    index:       int                    = 0                           # Ordinal position of port, according to legacy Scaffold.add_brick()
    minimum:     int                    = 1                           # For input ports, indicates how many connections are required for the brick to function correctly. 0 means not required. Ignored for output ports.
    maximum:     int                    = 1                           # For input ports, indicates how many connections total are permitted. 0 means unlimited. Ignored for output ports.
    channels:    dict[str, ChannelSpec] = field(default_factory=dict) # key is channel name

@dataclass
class ChannelData:
    """
        A bundle of neuron instances that work together to convey a specific topic of information.
        These are used to make connections at construction time.
    """
    spec:    ChannelSpec                             # This could be a different object than the one returned by brick reflections functions.
    neurons: list[str] = field(default_factory=list) # Graph node key for each neuron in this channel.

@dataclass
class PortData:
    """
        A collection of all the neuron instances associated with a given input or output
        of a brick. These are organized into topical groups called channels.
    """
    spec:     PortSpec                                             # This could be a different object than the one returned by brick reflections functions. The PortSpec.channels field should be ignored. Get ChannelSpec info directly from ChannelData.spec.
    channels: dict[str, ChannelData] = field(default_factory=dict) # key is channel name

class PortError(Exception):
    pass

class PortUtil:
    @classmethod
    def autoport_match(cls, port_name, query):
        """
            Determines if query is a legitimate auto-port name derived from port_name.
            Also returns true if query exactly matches port_name. In that case, it's
            not actually an auto-port, just a regular port name.
            (We could add an option for strict, in which case this would only return
            true if the query has a numeric suffix.)
        """
        if not query.startswith(port_name): return False
        prefix = len(port_name)
        if len(query) == prefix: return True
        return query[prefix:].isdigit()

    @classmethod
    def find_port_name(cls, ports: dict[str, PortSpec], index: int):
        """
        Find the port name corresponding to the given position index.
        If no such port exists, the given index is returned as a string (assumes old-style bricks).
        This method assumes that the dictionary keys exactly match the PortSpec.name
        of their associated values.
        """
        for p in ports.values():
            if p.index == index: return p.name
        if ports and index > 0 and p.spec.maximum != 1:  # Check for auto-port. Only works when there is one auto-port and it is in last position.
            return p.name + str(index - len(ports) + 2)  # adjust index to be one-based relative to last entry in ports
        return str(index)

    @classmethod
    def find_port_index(cls, ports: dict[str, PortSpec], name: str):
        """
        Find the position index of the port with the given name.
        If no such port exists, the given name is returned as an int (assumes old-style bricks).
        This method assumes that the dictionary keys exactly match the PortSpec.name
        of their associated values.
        """
        port = ports.get(name)
        if port: return port.index
        try:
            return int(name)  # For old-style bricks.
        except ValueError:  # This implies that name has some alpha characters, so try auto-port matching instead.
            for port_name, port in ports.items():
                if cls.autoport_match(port_name, name): return int(name[len(port_name):]) - 1 + port.index
        raise PortError("Can't find match for port name.")

    @classmethod
    def make_ports_from_specs(cls, specs: dict[str, PortSpec]) -> dict[str, PortData]:
        result = {}
        for port_name, port_spec in specs.items():
            port_data = PortData(spec=port_spec)
            result[port_name] = port_data
            for channel_name, channel_spec in port_spec.channels.items():
                port_data.channels[channel_name] = ChannelData(spec=channel_spec)
        return result

    @classmethod
    def get_autoports(cls, ports: dict[str, PortData], autoport_name: str = 'input', count: int = 1):
        """
        Returns a tuple of ports matching given name.
        count (int): Size of the tuple to return. Passing zero for count cause the tuple to contain
            all ports matching the name.
        """
        result = ()
        for port_name, port in ports.items():
            if not cls.autoport_match(autoport_name, port_name): continue
            result = (*result, port)
            if count and len(result) == count: break
        return result
