# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister, CircuitError
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, Statevector

from qiskit_addon_aqc_tensor import parametrize_circuit

# pylint: disable=no-self-use


class TestParametrizeCircuit:
    def test_parametrize_from_random_circuit_process_fidelity(self):
        qc = random_circuit(6, 12, max_operands=3, seed=7994855845011355715)
        ansatz, initial_params = parametrize_circuit(qc)
        ansatz.assign_parameters(initial_params, inplace=True)
        np.testing.assert_allclose(Operator(ansatz), Operator(qc))

    def test_parametrize_from_random_circuit_state_fidelity(self):
        qc = random_circuit(6, 12, max_operands=3, seed=4692760228210974079)
        ansatz, initial_params = parametrize_circuit(qc)
        ansatz.assign_parameters(initial_params, inplace=True)
        np.testing.assert_allclose(Statevector(ansatz), Statevector(qc))

    def test_parametrize_circuit_with_parameters(self):
        qubits = QuantumRegister(3)
        qc = QuantumCircuit(qubits)
        alpha1 = Parameter("alpha1")
        alpha2 = Parameter("alpha2")

        qc.ry(alpha1, [0])
        qc.rz(0.1, [0])
        qc.ry(alpha2, [1])
        qc.rz(alpha1, [1])
        qc.ry(0.2, [2])
        qc.rz(0.3, [2])

        ansatz, initial_params = parametrize_circuit(qc)
        ansatz.assign_parameters(
            {
                param: val
                for param, val in zip(ansatz.parameters, initial_params)
                if val is not None
            },
            inplace=True,
        )

        qc.assign_parameters([0.4, 0.5], inplace=True)
        ansatz.assign_parameters([0.4, 0.5], inplace=True)

        np.testing.assert_allclose(Operator(ansatz), Operator(qc))

    def test_parameter_conflict(self):
        qubits = QuantumRegister(1)
        qc = QuantumCircuit(qubits)
        theta0 = Parameter("theta[0]")
        qc.ry(theta0, [0])
        qc.ry(0.1, [0])
        with pytest.raises(CircuitError, match="conflict"):
            _ = parametrize_circuit(qc)