# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
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
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_addon_aqc_tensor.simulation import (
    tensornetwork_from_circuit,
)


class TestExactStatevector:
    def test_bell_circuit_statevector(self, bell_qc, available_backend_fixture):
        out_state = np.zeros([4], dtype=complex)
        tensornetwork_from_circuit(bell_qc, available_backend_fixture, out_state=out_state)
        assert out_state == pytest.approx(np.array([1, 0, 0, 1]) / np.sqrt(2))

    def test_random_circuit_statevector(self, available_backend_fixture):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.ryy(0.3, 1, 2)
        qc.rxx(0.7, 0, 2)
        out_state = np.zeros([8], dtype=complex)
        tensornetwork_from_circuit(qc, available_backend_fixture, out_state=out_state)
        abs(np.vdot(Statevector(qc).data, out_state)) == pytest.approx(1)
