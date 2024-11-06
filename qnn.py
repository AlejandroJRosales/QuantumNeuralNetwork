import numpy as np
import matplotlib.pyplot as plt
# dataset libs
from qiskit_algorithms.utils import algorithm_globals
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
# add qml libs here
from qiskit_machine_learning.neural_networks import SamplerQNN


class QNN:
    def __init__(self, data=None) -> None:
        self.data = data
        self.weight_dim = 4
        self.circut_input = ParameterVector("input", 2)
        self.circut_weights = ParameterVector("weight", 4)

        # define and set circut
        self.qc = QuantumCircuit(2)
        self.build_circut()
        
        # set inputs and weights
        self.input = data.input
        self.weights = algorithm_globals.random.random(self.weight_dim)

        self.qnn_sampler = SamplerQNN(circuit=self.qc, input_params=self.circut_input, weight_params=self.circut_weights)

    def forward(self):
        self.forward_out = self.qnn_sampler.forward(self.input, self.weights)

    def backward(self):
        self.input_grad, self.weight_grad = self.qnn_sampler.backward(self.input, self.weights)

    def build_circut(self):
        self.qc.ry(self.circut_input[0], 0)
        self.qc.ry(self.circut_input[1], 1)
        self.qc.cx(0, 1)
        self.qc.ry(self.circut_weights[0], 0)
        self.qc.ry(self.circut_weights[1], 1)
        self.qc.cx(0, 1)
        self.qc.ry(self.circut_weights[2], 0)
        self.qc.ry(self.circut_weights[3], 1)


class Data:
    def __init__(self):
        # initialize seed for reproducibility
        algorithm_globals.random_seed = 12345

        self.input_dim = 2
        self.input = algorithm_globals.random.random(self.input_dim)

# init data class for qnn
rand_data = Data()
# init qnn class with data
qnn = QNN(data=rand_data)
# run non-batched forward pass
qnn.forward()
print(qnn.forward_out, qnn.forward_out.shape)
# run backward pass
qnn.backward()
print(qnn.weight_grad, qnn.weight_grad.shape)

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