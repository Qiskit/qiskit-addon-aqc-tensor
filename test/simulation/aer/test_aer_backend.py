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
from qiskit.circuit.library import CXGate, XGate
from qiskit.providers.basic_provider import BasicSimulator

from qiskit_addon_aqc_tensor.simulation import (
    compute_overlap,
    tensornetwork_from_circuit,
)
from qiskit_addon_aqc_tensor.simulation.abstract import (
    _compute_overlap_with_local_gate_applied,
)
from qiskit_addon_aqc_tensor.simulation.aer import (
    QiskitAerSimulationSettings,
    is_aer_available,
)

skip_aer_tests = not is_aer_available()
pytestmark = pytest.mark.skipif(skip_aer_tests, reason="qiskit-aer is not installed")


class TestQiskitAerBackend:
    def test_bell_circuit(self, bell_qc, AerSimulator):
        simulator = AerSimulator(method="matrix_product_state")
        settings = QiskitAerSimulationSettings(simulator)
        bell_mps1 = tensornetwork_from_circuit(bell_qc, settings)
        bell_mps2 = tensornetwork_from_circuit(bell_qc, settings)
        assert compute_overlap(bell_mps1, bell_mps2) == pytest.approx(1)

    def test_bell_circuit_statevector(self, bell_qc, AerSimulator):
        simulator = AerSimulator(method="matrix_product_state")
        out_state = np.zeros([4], dtype=complex)
        tensornetwork_from_circuit(bell_qc, simulator, out_state=out_state)
        assert out_state == pytest.approx(np.array([1, 0, 0, 1]) / np.sqrt(2))

    def test_bell_circuit_log(self, bell_qc, AerSimulator):
        simulator = AerSimulator(method="matrix_product_state", mps_log_data=True)
        all_log_data: list[str] = []

        def my_callback(_, result):
            metadata = result.results[0].metadata
            mps_log_data = metadata["MPS_log_data"]
            all_log_data.append(mps_log_data)

        settings = QiskitAerSimulationSettings(simulator, callback=my_callback)
        tensornetwork_from_circuit(bell_qc, settings)
        assert len(all_log_data) == 1

    def test_invalid_simulator_choice(self, bell_qc):
        simulator = BasicSimulator()
        settings = QiskitAerSimulationSettings(simulator)
        with pytest.raises(TypeError) as e_info:
            tensornetwork_from_circuit(bell_qc, settings)
        assert e_info.value.args[0] == "simulator must be of type AerSimulator."

    def test_invalid_simulator_settings(self, bell_qc, AerSimulator):
        with pytest.raises(ValueError) as e_info:
            simulator = AerSimulator(method="automatic")
            tensornetwork_from_circuit(bell_qc, simulator)
        assert (
            e_info.value.args[0]
            == "AerSimulator must be configured to use 'matrix_product_state' method."
        )

    def test_invalid_out_state(self, bell_qc, subtests, AerSimulator):
        simulator = AerSimulator(method="matrix_product_state")
        with subtests.test("Invalid out_state type"):
            out_state = list(np.zeros([4], dtype=complex))
            with pytest.raises(TypeError) as e_info:
                tensornetwork_from_circuit(bell_qc, simulator, out_state=out_state)
            assert e_info.value.args[0] == "If provided, `out_state` must be of type numpy.ndarray."

        with subtests.test("Invalid out_state length"):
            out_state = np.zeros([6], dtype=complex)
            with pytest.raises(ValueError) as e_info:
                tensornetwork_from_circuit(bell_qc, simulator, out_state=out_state)
            assert (
                e_info.value.args[0]
                == "If provided, `out_state` must have size 2**num_qubits (4), not 6."
            )

    def test_invalid_mps(self, bell_qc, AerSimulator):
        simulator = AerSimulator(method="matrix_product_state")
        settings = QiskitAerSimulationSettings(simulator)
        bell_mps_1 = tensornetwork_from_circuit(bell_qc, settings)
        bell_mps_2 = tensornetwork_from_circuit(bell_qc, settings)
        bell_mps_2.lamb.append(1)
        with pytest.raises(RuntimeError) as e_info:
            compute_overlap(bell_mps_1, bell_mps_2)
        assert e_info.value.args[0] == "The lambda matrix array has an unexpected length."

    def test_incompatible_mps_overlaps(self, bell_qc, ghz_qc, AerSimulator):
        simulator = AerSimulator(method="matrix_product_state")
        settings = QiskitAerSimulationSettings(simulator)
        bell_mps_1 = tensornetwork_from_circuit(bell_qc, settings)
        ghz_mps = tensornetwork_from_circuit(ghz_qc, settings)
        with pytest.raises(ValueError) as e_info:
            compute_overlap(bell_mps_1, ghz_mps)
        assert (
            e_info.value.args[0]
            == "The matrix-product states have different numbers of qubits (2 vs. 3)."
        )

    def test_invalid_overlap_after_gate_application(self, bell_qc, AerSimulator):
        simulator = AerSimulator(method="matrix_product_state")
        settings = QiskitAerSimulationSettings(simulator)
        bell_mps_1 = tensornetwork_from_circuit(bell_qc, settings)
        bell_mps_2 = tensornetwork_from_circuit(bell_qc, settings)
        with pytest.raises(IndexError) as e_info:
            _compute_overlap_with_local_gate_applied(bell_mps_1, XGate(), 2, bell_mps_2)
        assert e_info.value.args[0] == "Invalid qubit index for 2 qubits: 2"

    def test_invalid_gate_application(self, bell_qc, AerSimulator):
        simulator = AerSimulator(method="matrix_product_state")
        settings = QiskitAerSimulationSettings(simulator)
        bell_mps_1 = tensornetwork_from_circuit(bell_qc, settings)
        bell_mps_2 = tensornetwork_from_circuit(bell_qc, settings)
        with pytest.raises(ValueError) as e_info:
            _compute_overlap_with_local_gate_applied(bell_mps_1, CXGate(), 1, bell_mps_2)
        assert e_info.value.args[0] == "The gate must act on a single qubit."
