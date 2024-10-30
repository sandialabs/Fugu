import abc
import sys
from abc import abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta("ABC", (), {"__slots__": ()})


class BrickTest(ABC):
    def setup_method(self):
        self.backend = None
        self.backend_args = {}
        self.debug = False

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
        if self.debug:
            scaffold.summary(verbose=2)
        self.backend.compile(scaffold, self.backend_args)
        if self.debug:
            print("Running circuit for {} timesteps".format(timesteps))
        return self.backend.run(timesteps)

    def basic_test(self, input_values, expected_output):
        timesteps = self.calculate_max_timesteps(input_values)
        scaffold = self.build_scaffold(input_values)
        spikes = self.run_scaffold(scaffold, timesteps)
        self.check_spike_output(spikes, expected_output, scaffold)

    def run_property_test(self, initial_values, new_properties, expected_outputs):
        scaffold = self.build_scaffold(initial_values)
        if self.debug:
            scaffold.summary(verbose=2)

        timesteps = self.calculate_max_timesteps(initial_values)
        if self.debug:
            print("Running for {} timesteps".format(timesteps))

        self.backend.compile(scaffold, self.backend_args)

        before_results = self.backend.run(timesteps)

        for properties, output in zip(new_properties, expected_outputs):
            if self.debug:
                print("Property test {}".format(properties))
            self.backend.reset()

            self.backend.set_properties(properties)

            after_results = self.backend.run(timesteps)

            self.check_spike_output([before_results, after_results], output, scaffold)

    def teardown_method(self):
        self.backend.cleanup()
        self.debug = False
