import numpy as np
import matplotlib.pyplot as plt
# dataset libs
from qiskit_algorithms.utils import algorithm_globals
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
# add qml libs here
from qiskit_machine_learning.neural_networks import SamplerQNN


class QNN:
    def __init__(self, data=None, qc=None) -> None:
        self.data = data
        self.qc = qc
        
        # set inputs and weights
        self.input = data.input
        self.weight_dim = 4
        self.weights = algorithm_globals.random.random(self.weight_dim)

        self.qnn_sampler = SamplerQNN(circuit=self.qc.qc, input_params=self.qc.input, weight_params=self.qc.weights)

    def forward(self):
        self.qnn_sampler_forward = self.qnn_sampler.forward(self.input, self.weights)

    def backward(self):
        self.qnn_sample_input_grad, self.qnn_sampler_weight_grad = self.qnn_sampler.backward(self.input, self.weights)


class Data:
    def __init__(self):
        # initialize seed for reproducibility
        algorithm_globals.random_seed = 12345

        self.input_dim = 2
        self.input = algorithm_globals.random.random(self.input_dim)


class Circuit:
    def __init__(self) -> None:
        self.qc = QuantumCircuit(2)
        self.input = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)
        self.qc.ry(self.input[0], 0)
        self.qc.ry(self.input[1], 1)
        self.qc.cx(0, 1)
        self.qc.ry(self.weights[0], 0)
        self.qc.ry(self.weights[1], 1)
        self.qc.cx(0, 1)
        self.qc.ry(self.weights[2], 0)
        self.qc.ry(self.weights[3], 1)

data = Data()
# define and set qc
qc = Circuit()
qnn = QNN(data, qc)
qnn.forward()
print(qnn.qnn_sampler_forward, qnn.qnn_sampler_forward.shape)


# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.