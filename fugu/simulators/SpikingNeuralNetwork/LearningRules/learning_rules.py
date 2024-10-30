from abc import ABC, abstractmethod

import numpy as np


class LearningRule(ABC):
    """
    Abstract base class for learning rules. All learning rules should inherit from this class
    """

    def __init__(self, name: str = None, three_factor: bool = False) -> None:
        """
        Constructor for the base learning class

        Parameters:
            name (str): Name of the learning rule
            three_factor (bool): Whether the learning rule is three factor or a simple homosynaptic rule

        Returns:
            None
        """
        self.name = name
        self.three_factor = three_factor
        self.weight_trace = []

    @abstractmethod
    def update_weights(self):
        """
        Abstract method to update weights based on the learning rule
        """
        pass


class STDPupdate(LearningRule):
    """
    The STDP learning rule class. Inherits from the Learning_Rules class. The STDP learning rule
    is a simple homosynaptic learning rule that updates weights based on the relative timing of
    pre and post synaptic spikes. The update rule is given by:
    dw = A_p * exp((pre_spike - post_spike) / tau) if post_spike - pre_spike >= 0
    dw = A_n * exp((post_spike - pre_spike) / tau) if post_spike - pre_spike < 0
    """

    def __init__(self, name=None, weight=None, A_p: float = 0.01, A_n: float = -0.01, tau: float = 0.001, time_step: int = 1):
        """
        Constructor for the STDP learning rule class. Inherits from the Learning_Rules class

        Parameters:
            name(any): String, optional. String name of a neuron. Default is None
            weight(float or list): Float or list, optional. The weight of the synapse. Default is None
            A_p(float): Float, optional. The potentiation factor. Default is 0.01
            A_n(float): Float, optional. The depression factor. Default is -0.01
            tau(float): Float, optional. The time constant. Default is 0.001
            time_step(int): Integer, optional. The time step. Default is 1
        """
        # TODO: Add validation conditions for the parameters

        self.name = name
        self._w = weight
        self._A_p = A_p
        self._A_n = A_n
        self._tau = tau
        self.time_step = time_step

    def update_weights(self, pre_spike_hist: list, post_spike_hist: list) -> float:
        """
        Update weights based on the STDP learning rule
        """

        # Check if a pre-synaptic spike has occured and keep track of its timing
        if pre_spike_hist[-1] == 1:
            pre_spike = self.time_step
            pre_status = 1
        else:
            pre_spike = self.calculate_spike_timing(pre_spike_hist)

        # Calculate the timing of the post synaptic spike
        if post_spike_hist[-1] == 1:
            post_spike = self.time_step
            post_status = 1
        else:
            post_spike = self.calculate_spike_timing(post_spike_hist)

        # If the post synaptic spike occurs before the pre synaptic spike, reduce the weight
        if post_spike - pre_spike < 0 and pre_status == 1:
            dw = self._A_n * np.exp((post_spike - pre_spike) / self._tau)
            self._w += dw
            post_status = 0

        # If the post synaptic spike occurs after or at the same time as the pre synaptic spike, increase the weight
        elif post_spike - pre_spike >= 0 and post_status == 1:
            dw = self._A_p * np.exp((pre_spike - post_spike) / self._tau)
            self._w += dw
            post_status = 0

        # Else keep it unchanged
        else:
            dw = 0
        self.weight_trace.append(self._w)

        return self._w

    @staticmethod
    def calculate_spike_timing(spike_hist: list) -> int:
        """
        Calculate the timing between the reference and either pre or post synaptic spike

        Parameters:
            spike_hist (deque): Spike history of the post synaptic neuron

        Returns:
            int: delay between the pre and post synaptic spikes
        """
        spike_ind = max([ind for ind, val in enumerate(spike_hist) if val])
        spike_time = len(spike_hist) - spike_ind
        return spike_time


class RSTDPupdate(LearningRule):

    def __init__(self, name=None, weight=None, reward: float = 0.01, A_p: float = 0.01, A_n: float = -0.01, tau: float = 0.001, time_step: int = 1):
        """
        Constructor for the R-STDP learning rule class. Inherits from the Learning_Rules class

        Parameters:
                name(any): String, optional. String name of a neuron. Default is None
                weight(float or list): Float or list, optional. The weight of the synapse. Default is None
                reward(float): Float, optional. The reward factor to scale the eligibility trace. Default is 0.01
                A_p(float): Float, optional. The potentiation factor. Default is 0.01
                A_n(float): Float, optional. The depression factor. Default is -0.01
                tau(float): Float, optional. The time constant for potentiation. Default is 0.001
                time_step(int): Integer, optional. The time step. Default is 1

        """

        self.name = name
        self._w = weight
        self._A_p = A_p
        self._A_n = A_n
        self._tau = tau
        self._reward = reward
        self.time_step = time_step
        self.base_update_obj = STDPupdate(weight=self._w, A_p=self._A_p, A_n=self._A_n, tau=self._tau, time_step=self.time_step)
        self.eligibility_trace = 0.0

    def update_weights(self, pre_spike_hist: list, post_spike_hist: list):
        """
        Update weights based on the r-stdp learning rule
        e = -(e/tau) + STDP(pre, post)
        dw = reward * e
        """
        # Getting the STDP update value
        base_update = self.base_update_obj.update_weights(pre_spike_hist, post_spike_hist)

        # Update the eligibility trace
        self.eligibility_trace = -self.eligibility_trace / self._tau + base_update

        # Update the weight as a function of the reward and the eligibility trace
        self._w += self._reward * self.eligibility_trace


# class STDP_update():

#     def __init__(self, A_p, A_n, tau, time_step):
#         self.A_p = A_p
#         self.A_n = A_n
#         self.tau = tau
#         self.time_step = time_step

#     def __call__(self, w: np.ndarray, pre_spike: np.ndarray, post_spike: np.ndarray):

#         for weight in range(w.size):
#             if pre_spike[weight] == 1:
#                 pre_spike = self.time_step
#                 post_status = 1
#             pre_index[self.time_step + 1] = pre_spike
#             post_index[self.time_step + 1] = post_spike
#             if post_spike - pre_spike < 0 and post_status == 1:
#                 dw[weight][self.time_step + 1] = A_n * np.exp((post_spike - pre_spike) / tau)
#                 w[weight] += dw[weight][self.time_step + 1]
#                 post_status = 0
#             elif post_spike - pre_spike >= 0:
#                 dw[weight][self.time_step + 1] = A_p * np.exp((pre_spike - post_spike) / tau)
#                 w[weight] += dw[weight][self.time_step + 1]
#                 post_status = 0
#             else:
#                 dw[weight][self.time_step + 1] = 0


class AveragingTraceRule:

    def __init__(self, Npre, Npos, beta1, beta2, beta3, base_confidence_val, trace_threshold, trace_scaler, lr, lr_decay):
        self.beta1 = beta1
        self.beta2 = beta2
        self.Npre = Npre
        self.Npos = Npos
        self.threshold = trace_threshold
        self.trace_scaler = trace_scaler
        self.base_confidence_val = base_confidence_val
        self.lr_base = lr
        self.lr_decay = lr_decay

    def __call__(self, W, pre, post, mod, trace_prev, curr_activ):
        """
        Trace(t) = Base_conf_val + (activ_curr/a_mean)*Trace(t-1)
        This uses the general dynamic learning rule ( The clamping can also be added to the weight update )
        :param trace_prev: [Nhidden x 1] The previous trace value of the selected neuron
        :param curr_activ: [Nhidden x 1] The current activation values in the hidden layer
        :return: W: The updated weights
        :return: T: The modified trace
        """

        dW = np.zeros_like(W)

        activ_mean = np.mean(curr_activ)
        # Modify the trace value
        dT = self.trace_scaler * ((curr_activ / activ_mean) * trace_prev)
        # Update the trace
        T = self.base_confidence_val + dT

        for j in range(self.Npos):
            # Modify the learning rate based on the updated trace
            if T[j] > self.threshold:
                self.lr = self.lr_base - self.lr_decay
            else:
                self.lr = self.lr_base
            dW[j, :] = self.lr * mod[j] * (pre * (post[j] - self.beta1) - self.beta2 * (post[j] - self.beta1))
        W = W + dW
        return W, T


class MovingAverageTraceRule:

    def __init__(self, Npre, Npos, beta1, beta2, beta3, base_confidence_val, trace_threshold, trace_scaler, lr, lr_decay):
        self.beta1 = beta1
        self.beta2 = beta2
        self.Npre = Npre
        self.Npos = Npos
        self.threshold = trace_threshold
        self.trace_scaler = trace_scaler
        self.base_confidence_val = base_confidence_val
        self.lr_base = lr
        self.lr_decay = lr_decay

    def __call__(self, W, pre, post, mod, trace_prev, curr_activ, CMA_prev, sample_count):
        """
        Trace(t) = Base_conf_val*Trace(t-1) + CMA(curr_activ)*alpha
        This uses the general dynamic learning rule ( The clamping can also be added to the weight update )
        :param trace_prev: [Nhidden x 1] The previous trace value of the selected neuron
        :param curr_activ: [Nhidden x 1] The current activation values in the hidden layer
        :return: W: The updated weights
        :return: T: The modified trace
        """
        dW = np.zeros_like(W)
        # Modify the trace value
        CMA_new = CMA_prev + (curr_activ - CMA_prev) / (sample_count + 1)
        # Sign parameter that triggers based on network response and the current activation value
        alpha = np.ones_like(curr_activ)
        alpha[curr_activ < 0.75] = -1
        T = self.base_confidence_val * trace_prev + alpha * CMA_new

        for j in range(self.Npos):

            # Modify the learning rate based on the updated trace
            if T[j] > self.threshold:
                self.lr = self.lr_base - self.lr_decay
            else:
                self.lr = self.lr_base
            dW[j, :] = self.lr * mod[j] * (pre * (post[j] - self.beta1) - self.beta2 * (post[j] - self.beta1))
        W = W + dW
        return W, T


class ExponentialMovingAverageTraceRule:
    """
    sample_count: Start the sample count with 0 and keep on updating while changing the trace
    """

    def __init__(self, Npre, Npos, beta1, beta2, beta3, base_confidence_val, trace_threshold, trace_scaler, lr, lr_decay, sample_count, activ_arr):
        self.beta1 = beta1
        self.beta2 = beta2
        self.Npre = Npre
        self.Npos = Npos
        self.threshold = trace_threshold
        self.trace_scaler = trace_scaler
        self.base_confidence_val = base_confidence_val
        self.lr_base = lr
        self.sample_count = sample_count
        self.lr_decay = lr_decay
        self.activ_arr = activ_arr

    def __call__(self, W, pre, post, mod, trace_prev, curr_activ, CMA_prev):
        """
        Trace(t) = Base_conf_val*Trace(t-1)*(1/std(current_activ)*exp(-1/2*(current_activ-mean(current_activ)/std(a)))
         + CMA(curr_activ)*alpha
        This uses the general dynamic learning rule ( The clamping can also be added to the weight update )
        :param trace_prev: [Nhidden x 1] The previous trace value of the selected neuron
        :param curr_activ: [Nhidden x 1] The current activation values in the hidden layer
        :return: W: The updated weights
        :return: T: The modified trace
        """
        self.sample_count += 1

        if self.sample_count == 100:
            self.sample_count = 0
            self.activ_arr = []
            self.activ_arr.append(curr_activ)
        self.activ_arr.append(curr_activ)
        temp_activ_arr = np.array(self.activ_arr)
        temp = -0.5 * ((curr_activ - np.mean(temp_activ_arr)) / np.std(temp_activ_arr))
        gauss_factor = 1 / (np.std(temp_activ_arr)) * (np.exp(temp))
        dW = np.zeros_like(W)
        # Modify the trace value
        CMA_new = CMA_prev + (curr_activ - CMA_prev) / (self.sample_count + 1)
        # Sign parameter that triggers based on network response and the current activation value
        alpha = np.ones_like(curr_activ)
        alpha[curr_activ < 0.75] = -1
        T = self.base_confidence_val * trace_prev * gauss_factor + alpha * CMA_new

        for j in range(self.Npos):

            # Modify the learning rate based on the updated trace
            if T[j] > self.threshold:
                self.lr = self.lr_base - self.lr_decay
            else:
                self.lr = self.lr_base
            dW[j, :] = self.lr * mod[j] * (pre * (post[j] - self.beta1) - self.beta2 * (post[j] - self.beta1))
        W = W + dW
        return W, T
