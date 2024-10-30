from dataclasses import dataclass


@dataclass
class LearningParams:
    """Parameters for learning.

    learning_rule: str: The learning rule to be selected from the available options. Currently from "STDP, r-STDP and None"
    A_p: float: The potentiation constant for the STDP learning rule
    A_n: float: The depression constant for the STDP learning rule
    tau: float: The time constant for the STDP learning rule
    bins: int: The number of time steps for the simulation window
    dt: float: The time step for the simulation
    Rm: float: The membrane resistance of the neuron
    Cm: float: The membrane capacitance of the neuron
    Tm: float: The membrane time constant of the neuron necessary for update calculation
    X_tar: float: The target value for the trace based learning rule
    """

    learning_rule: str = "STDP"
    A_p: float = 0.01
    A_n: float = -0.01
    tau: float = 100
    bins: int = 300
    dt: float = 0.001
    Rm: float = 1e8
    Cm: float = 1e-7
    Tm: float = Rm * Cm
    X_tar: float = 0.5


if __name__ == "__main__":
    learn_params = LearningParams()
    updated_params = LearningParams(A_p=0.02, A_n=-0.02, tau=200)
    print(learn_params, "\n", updated_params)
    print(learn_params.A_p, updated_params.A_p)
    data_keys = list(learn_params.__dict__)
    data_dict = learn_params.__dict__
    print(data_dict)
    print(data_keys, len(data_keys))
    # for i in range(len(data_keys)):
    #     print (learn_params.data_keys[i])
