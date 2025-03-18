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
r"""Gradient implementation for Aer backend."""
from __future__ import annotations

from typing import Union

import numpy as np
from plum import dispatch
from qiskit.circuit import (
    QuantumCircuit,
)

from ...objective import OneMinusFidelity
from ..abstract import (
    apply_circuit_to_state,
    compute_overlap,
    tensornetwork_from_circuit,
)
from ..explicit_gradient import (
    compute_gradient_of_tensornetwork_overlap,
    preprocess_circuit_for_backtracking,
)
from .simulation import AerSimulator, QiskitAerSimulationSettings


@dispatch
def _preprocess_for_gradient(objective, settings: Union[QiskitAerSimulationSettings, AerSimulator]):
    if objective._ansatz is not None:
        ansatz_num_qubits = objective._ansatz.num_qubits
        target_num_qubits = len(objective._target_tensornetwork.gamma)
        if ansatz_num_qubits != target_num_qubits:
            raise ValueError(
                "Ansatz and target have different numbers of qubits "
                f"({ansatz_num_qubits} vs. {target_num_qubits})."
            )
    gate_actions = preprocess_circuit_for_backtracking(objective._ansatz, settings)
    lhs_tensornetwork = tensornetwork_from_circuit(
        QuantumCircuit(objective._ansatz.num_qubits), settings
    )  # |0>
    return gate_actions, lhs_tensornetwork


@dispatch
def _compute_objective_and_gradient(
    objective: OneMinusFidelity,
    settings: Union[QiskitAerSimulationSettings, AerSimulator],
    preprocess_info,
    x: np.ndarray,
) -> tuple[float, np.ndarray]:
    gate_actions, lhs_tensornetwork = preprocess_info
    bound_qc = objective._ansatz.assign_parameters(x)
    vdagger_target = apply_circuit_to_state(
        bound_qc.inverse(), objective._target_tensornetwork, settings
    )
    overlap = compute_overlap(lhs_tensornetwork, vdagger_target)
    objective_value = 1 - abs(overlap) ** 2
    overlap_gradient = compute_gradient_of_tensornetwork_overlap(
        gate_actions=gate_actions,
        thetas=x,
        lhs=lhs_tensornetwork,
        vdagger_rhs=vdagger_target,
    )
    gradient = -2 * np.real(np.conj(overlap) * overlap_gradient)
    return objective_value, gradient
