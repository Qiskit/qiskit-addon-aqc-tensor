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
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import CXGate, RXGate, XXPlusYYGate

from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
from qiskit_addon_aqc_tensor.simulation import (
    compute_overlap,
    tensornetwork_from_circuit,
)
from qiskit_addon_aqc_tensor.simulation.abstract import (
    _apply_one_qubit_gate_inplace,
    _apply_two_qubit_gate_inplace,
)
from qiskit_addon_aqc_tensor.simulation.quimb import (
    QuimbSimulator,
    is_quimb_available,
    qiskit_ansatz_to_quimb,
    recover_parameters_from_quimb,
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
    def test_parameter_expression(self, quimb):
        qc = QuantumCircuit(1)
        p = Parameter("x")
        qc.rx(1 - p, 0)
        val = np.random.default_rng().random()
        circ, ctx = qiskit_ansatz_to_quimb(qc, [val], quimb.tensor.Circuit)
        recovered = recover_parameters_from_quimb(circ, ctx)
        assert recovered == pytest.approx([val])

    def test_parameter_count_mismatch(self, quimb):
        qc = QuantumCircuit()
        with pytest.raises(ValueError) as e_info:
            qiskit_ansatz_to_quimb(qc, [1.0], quimb.tensor.Circuit)
        assert (
            e_info.value.args[0]
            == "1 parameter(s) were passed, but the circuit has 0 parameter(s)."
        )

    def test_multiple_parameter_gate(self, quimb):
        qc = QuantumCircuit(2)
        qc.append(XXPlusYYGate(Parameter("x")), (0, 1))
        qiskit_ansatz_to_quimb(qc, [np.pi / 2], quimb.tensor.CircuitMPS)

    def test_parameterexpression_multiple_parameters_failure_message(self, quimb):
        qc = QuantumCircuit(1)
        x = Parameter("x")
        y = Parameter("y")
        qc.rx(x + y, 0)
        qc.ry(x - y, 0)
        with pytest.raises(ValueError) as e_info:
            qiskit_ansatz_to_quimb(qc, [np.pi / 2, np.pi / 4], quimb.tensor.Circuit)
        assert e_info.value.args[0] == "Each expression cannot contain more than one Parameter"

    def test_nonlinear_parameterexpression_failure_message(self, quimb):
        qc = QuantumCircuit(1)
        x = Parameter("x")
        qc.rx(x**2, 0)
        with pytest.raises(ValueError) as e_info:
            qiskit_ansatz_to_quimb(qc, [np.pi / 8], quimb.tensor.Circuit)
        assert (
            e_info.value.args[0]
            == "The Quimb backend currently requires that each ParameterExpression must be in the form mx + b (not x**2).  Otherwise, the backend is unable to recover the parameter."
        )

    def test_repeated_parameter(self, quimb):
        qc = QuantumCircuit(1)
        x = Parameter("x")
        qc.rx(x, 0)
        qc.ry(x, 0)
        with pytest.raises(ValueError) as e_info:
            qiskit_ansatz_to_quimb(qc, [np.pi / 2], quimb.tensor.Circuit)
        assert (
            e_info.value.args[0]
            == "Parameter x cannot be repeated in circuit, else quimb will attempt to optimize each instance separately."
        )

    def test_unspecified_gradient_method(self, quimb):
        settings = QuimbSimulator(quimb.tensor.CircuitMPS)
        qc = QuantumCircuit(1)
        x = Parameter("x")
        qc.rx(x, 0)
        with pytest.raises(ValueError) as e_info:
            MaximizeStateFidelity(None, qc, settings)
        assert (
            e_info.value.args[0]
            == "Gradient method unspecified. Please specify an autodiff_backend for the QuimbSimulator object."
        )

    def test_recovery_num_parameters_mismatch_error(self, quimb):
        x = Parameter("x")
        y = Parameter("y")
        qc1 = QuantumCircuit(1)
        qc1.rx(1 - x, 0)
        qc2 = QuantumCircuit(1)
        qc2.rx(1 - x, 0)
        qc2.ry(1 - 0.5 * y, 0)
        circ1, _ = qiskit_ansatz_to_quimb(qc1, [0.5], quimb.tensor.Circuit)
        _, ctx2 = qiskit_ansatz_to_quimb(qc2, [0.5, 0.3], quimb.tensor.Circuit)
        with pytest.raises(ValueError) as e_info:
            recover_parameters_from_quimb(circ1, ctx2)
        assert (
            e_info.value.args[0]
            == "The length of the mapping in the provided QiskitQuimbConversionContext does not match the number of parametrized gates in the circuit (2 vs. 1)."
        )
