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
        # define and set circut
        self.qc = QuantumCircuit(2)
        self.circut_input = ParameterVector("input", 2)
        self.circut_weights = ParameterVector("weight", 4)
        self.set_circut()
        # set inputs and weights
        self.input = data.input
        self.kernel_dim = 4
        self.weights = algorithm_globals.random.random(self.kernel_dim)

        self.qnn_sampler = SamplerQNN(circuit=self.qc, input_params=self.circut_input, weight_params=self.circut_weights)

    def forward(self):
        self.qnn_sampler_forward = self.qnn_sampler.forward(self.input, self.weights)
        print(self.qnn_sampler_forward, self.qnn_sampler_forward.shape)

    def set_circut(self):
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

data = Data()
qnn = QNN(data=data)
qnn.forward()

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