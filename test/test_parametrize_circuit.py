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

import pytest
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, Statevector, process_fidelity, state_fidelity

from qiskit_addon_aqc_tensor import parametrize_circuit

# pylint: disable=no-self-use


class TestParametrizeCircuit:
    def test_parametrize_from_random_circuit_process_fidelity(self):
        qc = random_circuit(6, 12, max_operands=3, seed=7994855845011355715)
        ansatz, initial_params = parametrize_circuit(qc)
        ansatz.assign_parameters(initial_params, inplace=True)
        fidelity = process_fidelity(Operator(ansatz), Operator(qc))
        assert fidelity == pytest.approx(1)

    def test_parametrize_from_random_circuit_state_fidelity(self):
        qc = random_circuit(6, 12, max_operands=3, seed=4692760228210974079)
        ansatz, initial_params = parametrize_circuit(qc)
        ansatz.assign_parameters(initial_params, inplace=True)
        fidelity = state_fidelity(Statevector(ansatz), Statevector(qc))
        assert fidelity == pytest.approx(1)
