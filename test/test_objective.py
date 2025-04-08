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
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
from qiskit_addon_aqc_tensor.simulation import (
    tensornetwork_from_circuit,
)


# pylint: disable=no-self-use
class TestObjective:
    def test_ansatz_target_qubit_mismatch(self, available_backend_fixture):
        settings = available_backend_fixture
        target = tensornetwork_from_circuit(QuantumCircuit(2), settings)
        x = Parameter("x")
        ansatz = QuantumCircuit(1)
        ansatz.rx(x, 0)
        with pytest.raises(ValueError) as e_info:
            MaximizeStateFidelity(target, ansatz, settings)
        assert (
            e_info.value.args[0] == "Ansatz and target have different numbers of qubits (1 vs. 2)."
        )
