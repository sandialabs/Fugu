import numpy as np
import pytest

from fugu.backends.snn_backend import snn_Backend
from fugu.bricks import NoOps, Vector_Input
from fugu.scaffold import Scaffold


class Test_NoOpsBrick:
    def test_noops_delay(self):
        noops_delay = 100
        scaffold = Scaffold()
        scaffold_input = scaffold.add_brick(Vector_Input(np.array([1]), name="vector_input"))
        noops = scaffold.add_brick(NoOps(noops_delay=noops_delay), output=True)
        scaffold.connect(scaffold_input, noops)
        scaffold.lay_bricks()

        backend = snn_Backend()
        backend_args = {}
        backend_args["record"] = "all"
        backend.compile(scaffold, backend_args)
        snn_results = backend.run(noops_delay + 1)

        calculated = snn_results[snn_results["time"] == noops_delay].to_numpy()

        assert int(calculated[0, 0]) == noops_delay
        assert int(calculated[0, 1]) == 3
