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
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, Statevector, process_fidelity, state_fidelity

from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit_addon_aqc_tensor.ansatz_generation import KAK

# pylint: disable=no-self-use


class TestAnsatzGeneration:
    def test_ansatz_from_random_circuit_process_fidelity(self):
        qc = random_circuit(6, 4, max_operands=2)
        ansatz, initial_params = generate_ansatz_from_circuit(qc)
        ansatz.assign_parameters(initial_params, inplace=True)
        fidelity = process_fidelity(Operator(ansatz), Operator(qc))
        assert fidelity == pytest.approx(1)

    def test_ansatz_from_random_circuit_state_fidelity(self):
        qc = random_circuit(6, 4, max_operands=2)
        ansatz, initial_params = generate_ansatz_from_circuit(qc, qubits_initially_zero=True)
        ansatz.assign_parameters(initial_params, inplace=True)
        fidelity = state_fidelity(Statevector(ansatz), Statevector(qc))
        assert fidelity == pytest.approx(1)

    def test_ansatz_fails_given_three_qubit_gate(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.ccx(0, 1, 2)
        with pytest.raises(ValueError) as e_info:
            generate_ansatz_from_circuit(qc)
        assert (
            e_info.value.args[0]
            == "Only one- and two-qubit operations are allowed in the original circuit."
        )

    def test_explicit_3cnot_kak_implementation(self):
        rng = np.random.default_rng()
        params = rng.standard_normal(3)
        kak = KAK(params)
        comparison_qc = QuantumCircuit(2)
        comparison_qc.rxx(-2 * params[0], 0, 1)
        comparison_qc.ryy(-2 * params[1], 0, 1)
        comparison_qc.rzz(-2 * params[2], 0, 1)
        fidelity = process_fidelity(kak, comparison_qc.to_gate())
        assert fidelity == pytest.approx(1)

    def test_parameter_count_check(self):
        rng = np.random.default_rng()
        params = rng.standard_normal(2)
        with pytest.raises(ValueError) as e_info:
            KAK(params)
        assert e_info.value.args[0] == "Wrong number of parameters"

    def test_circuit_with_barrier(self):
        qc = random_circuit(6, 4, max_operands=2)
        qc.barrier()
        qc.compose(random_circuit(6, 4, max_operands=2), inplace=True)
        ansatz, _ = generate_ansatz_from_circuit(qc)
        assert "barrier" not in {inst.operation.name for inst in ansatz.data}

    def test_dynamic_circuit(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], True)):
            qc.x(0)
        with pytest.raises(ValueError) as e_info:
            generate_ansatz_from_circuit(qc)
        assert (
            e_info.value.args[0]
            == "Circuits which operate on classical bits are not yet supported."
        )

    def test_idle_qubit(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        _, initial_parameters = generate_ansatz_from_circuit(qc)
        assert len(initial_parameters) == 3

    def test_large_circuit_two_qubit_weyl_decomposition_error(self):
        qc = QuantumCircuit(50)

        for _ in range(10):
            for i in range(0, 50, 2):
                qc.cx(i, i + 1)
            for i in range(1, 49, 2):
                qc.cx(i, i + 1)

        _, _ = generate_ansatz_from_circuit(qc, qubits_initially_zero=True)
