import unittest
import abc
import sys

from abc import abstractmethod

from fugu import Scaffold

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})


class BrickTest(ABC):
    backend = None
    backend_args = {}

    @abstractmethod
    def build_scaffold(self, input_values):
        pass

    def calculate_max_timesteps(self, input_values):
        return 10

    def check_spike_output(self, spikes, expected, scaffold):
        pass

    def convert_input(self, input_values):
        pass

    def run_scaffold(self, scaffold, timesteps):
        self.backend.compile(scaffold, self.backend_args)
        return self.backend.run(timesteps)

    def basic_test(self, input_values, expected_output):
        scaffold = self.build_scaffold(input_values)
        timesteps = self.calculate_max_timesteps(input_values)
        spikes = self.run_scaffold(scaffold, timesteps)
        self.check_spike_output(spikes, expected_output, scaffold)

    def run_parameter_test(self, initial_values, new_parameters, expected_outputs):
        scaffold = self.build_scaffold(initial_values)
        timesteps = self.calculate_max_timesteps(initial_values)

        self.backend.compile(scaffold, self.backend_args)

        before_results = self.backend.run(timesteps)

        self.backend.reset()

        self.backend.set_parameters(new_parameters)

        after_results = self.backend.run(timesteps)

        self.check_spike_output([before_results, after_results], expected_outputs, scaffold)

    def tearDown(self):
        self.backend.cleanup()

    @classmethod
    def tearDownClass(self):
        self.backend.cleanup()
