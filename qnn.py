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
        # set inputs and weights
        self.inputs = data.inputs
        self.weights = data.weights

        # init quantum circut
        self.qc = QuantumCircuit(2)
        self.set_circut()

        sampler_qnn = SamplerQNN(circuit=self.qc, input_params=self.inputs, weight_params=self.weights)


    def set_circut(self):
        self.qc.ry(self.inputs[0], 0)
        self.qc.ry(self.inputs[1], 1)
        self.qc.cx(0, 1)
        self.qc.ry(self.weights[0], 0)
        self.qc.ry(self.weights[1], 1)
        self.qc.cx(0, 1)
        self.qc.ry(self.weights[2], 0)
        self.qc.ry(self.weights[3], 1)


class Data:
    def __init__(self):
        # initialize seed for reproducibility
        algorithm_globals.random_seed = 12345

        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)


data = Data()
qnn = QNN(data=data)

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