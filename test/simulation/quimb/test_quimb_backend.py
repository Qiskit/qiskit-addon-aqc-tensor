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


import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import CXGate, RXGate

from qiskit_addon_aqc_tensor.simulation import (
    _apply_one_qubit_gate_inplace,
    _apply_two_qubit_gate_inplace,
    compute_overlap,
    tensornetwork_from_circuit,
)
from qiskit_addon_aqc_tensor.simulation.quimb import (
    QuimbSimulator,
    is_quimb_available,
    qiskit_ansatz_to_quimb,
)

skip_quimb_tests = not is_quimb_available()
pytestmark = pytest.mark.skipif(skip_quimb_tests, reason="qiskit-quimb is not installed")


class TestQuimbBackend:
    def test_bell_circuit(self, quimb):
        settings = QuimbSimulator(quimb.tensor.CircuitMPS)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        mps = tensornetwork_from_circuit(qc, settings)
        assert compute_overlap(mps, mps) == pytest.approx(1)

    def test_operations(self, quimb):
        settings = QuimbSimulator(quimb.tensor.CircuitMPS)
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        psi = tensornetwork_from_circuit(qc, settings)
        _apply_one_qubit_gate_inplace(psi, RXGate(0.2), 1)
        _apply_two_qubit_gate_inplace(psi, CXGate(), 1, 2, settings)


class TestQuimbConversion:
    def test_parameter_expression(self):
        qc = QuantumCircuit(1)
        p = Parameter("x")
        qc.rx(1 - p, 0)
        qiskit_ansatz_to_quimb(qc, [0])
        # FIXME: finish this test
