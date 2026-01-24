import numpy as np

import fugu
from fugu import Scaffold
from fugu.backends import snn_Backend
from fugu.bricks import AND_OR, Vector_Input

if __name__ == "__main__":

    scaffold = Scaffold()
    input1 = scaffold.add_brick(Vector_Input(np.array([1, 0, 1, 0, 1]), coding="Raster", name="input1"))
    input2 = scaffold.add_brick(Vector_Input(np.array([1, 1, 0, 0, 1]), coding="Raster", name="input2"))
    AND = scaffold.add_brick(AND_OR(name="AND"), output=True)
    OR = scaffold.add_brick(AND_OR(mode="OR", name="OR"), output=True)

    scaffold.connect(input1, AND)
    scaffold.connect(input2, OR)
    scaffold.connect(AND, OR)
    scaffold.connect(OR, AND)
    scaffold.lay_bricks(verbose=0)
    # scaffold.summary(verbose=0)

    backend = snn_Backend()
    backend.compile(scaffold)
    output = backend.run(5)
    print(output)
