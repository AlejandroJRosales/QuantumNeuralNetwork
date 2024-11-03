import numpy as np
import matplotlib.pyplot as plt
# dataset libs
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.datasets import ad_hoc_data
# add qml libs here


class QNN:
    def __init__(self, data):
        self.data = data


class Data:
    def __init__(self):
        # initialize seed for reproducibility
        algorithm_globals.random_seed = 12345

        # initalize training and testing data
        self.adhoc_dimension = 2
        self.train_features, self.train_labels, self.test_features, self.test_labels, self.adhoc_total = ad_hoc_data(
            training_size=20,
            test_size=5,
            n=self.adhoc_dimension,
            gap=0.3,
            plot_data=False,
            one_hot=False,
            include_sample_total=True,
        )

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