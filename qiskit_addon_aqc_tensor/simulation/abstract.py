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

"""Abstract interface for a tensor-network backend."""

from __future__ import annotations

from typing import Optional

import numpy as np
from plum import dispatch
from qiskit.circuit import Gate, QuantumCircuit

# NOTE: This file contains abstract classes and functions.  The functions in
# this file are implemented differently for each tensor-network backend, and
# the backend method is chosen dynamically based on the type(s) passed to the
# function.  Dispatch is powered by plum-dispatch, a multiple dispatch library
# for Python.


class TensorNetworkState:
    """Abstract tensor network state."""


class TensorNetworkSimulationSettings:
    """Abstract truncation settings."""


###
# Basic operations
###


@dispatch.abstract
def compute_overlap(
    psi_1: TensorNetworkState, psi_2: TensorNetworkState, /
) -> complex:  # pragma: no cover
    r"""Compute overlap of two quantum states: :math:`\langle \psi_1 | \psi_2 \rangle`.

    NOTE: Unlike ``numpy.dot``, implementations of this method will perform
    complex conjugation on the first argument.

    Args:
        psi_1: first quantum state.
        psi_2: second quantum state.

    Returns:
        complex dot product value.
    """
    raise NotImplementedError


@dispatch.abstract
def tensornetwork_from_circuit(
    qc: QuantumCircuit, settings: TensorNetworkSimulationSettings, /
) -> TensorNetworkState:  # pragma: no cover
    """Create a tensor-network state by simulating a quantum circuit.

    The type of tensor-network state will correspond to the type of the
    ``settings`` object.  For instance, a :class:`.QiskitAerSimulationSettings`
    will result in this function returning a :class:`.QiskitAerMPS`.
    """
    raise NotImplementedError


@dispatch.abstract
def apply_circuit_to_state(
    qc: QuantumCircuit,
    psi: TensorNetworkState,
    settings: TensorNetworkSimulationSettings,
    /,
    *,
    out_state: Optional[np.ndarray] = None,
) -> TensorNetworkState:  # pragma: no cover
    """Apply a quantum circuit to a tensor network state.

    The input state (``psi``) is not modified.

    Returns:
        The new state.
    """
    raise NotImplementedError


###
# Generalized gradient interface
###


@dispatch.abstract
def _preprocess_for_gradient(objective, settings: TensorNetworkSimulationSettings):
    raise NotImplementedError


@dispatch.abstract
def _compute_objective_and_gradient(
    objective, settings: TensorNetworkSimulationSettings, preprocess_info, x: np.ndarray
) -> tuple[float, np.ndarray]:
    raise NotImplementedError


###
# Operations required for explicit gradient implementation.  If a backend
# implements these methods, it can then rely on the explicit gradient
# implementation.
###


@dispatch.abstract
def _compute_overlap_with_local_gate_applied(
    psi_1: TensorNetworkState, gate: Gate, qubit: int, psi_2: TensorNetworkState, /
) -> complex:  # pragma: no cover
    r"""Compute overlap of two quantum states with a single local (one-qubit) gate in the middle.

    :math:`\langle \psi_1 | G | \psi_2 \rangle`.

    NOTE: Unlike ``numpy.dot``, implementations of this method will perform
    complex conjugation on the first argument.

    Args:
        psi_1: first quantum state.
        gate: Gate to apply.
        qubit: index of the qubit where the local gate is applied.
        psi_2: second quantum state.

    Returns:
        complex dot product value.
    """
    raise NotImplementedError


@dispatch.abstract
def _apply_one_qubit_gate_inplace(
    psi: TensorNetworkState, gate: Gate, qubit: int, /
) -> None:  # pragma: no cover
    """Apply one-qubit gate in place."""
    raise NotImplementedError


@dispatch.abstract
def _apply_two_qubit_gate_inplace(
    psi: TensorNetworkState,
    gate: Gate,
    q0: int,
    q1: int,
    settings: TensorNetworkSimulationSettings,
    /,
) -> None:  # pragma: no cover
    """Apply two-qubit gate in place."""
    raise NotImplementedError
