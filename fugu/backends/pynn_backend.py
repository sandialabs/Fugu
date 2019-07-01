import os, sys
import pyNN
from fugu.backends.backend import Backend

class pynn_Backend(Backend):
    def stream(self,
               scaffold,
               input_values,
               stepping,
               record,
               backend_args):
        pass
    
    def batch(self,
             scaffold,
             input_values,
             n_steps,
             record,
             backend_args):
        pass
