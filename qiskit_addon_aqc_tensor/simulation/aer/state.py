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

"""Aer MPS state and basic operations that do not require an MPS simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from plum import dispatch
from qiskit.circuit import Gate

from ..abstract import TensorNetworkState


@dataclass
class QiskitAerMPS(TensorNetworkState):
    """Qiskit Aer representation of a matrix-product state.

    This form of a matrix-product state was introduced in the original
    time-evolving block decimation (TEBD) paper,
    https://arxiv.org/abs/quant-ph/0301063.
    See Sec. 7.3.2 of https://arxiv.org/abs/1008.3477v2 for more details on the relationship
    with other MPS representations.
    """

    #: Gamma matrices (list of 2-tuples of ndarrays, one for each site).
    gamma: list[tuple[np.ndarray, np.ndarray]]
    #: Lambda matrices (list of ndarrays, one for each bond between adjacent qubits).
    lamb: list[np.ndarray]

    def _as_tuple(self):
        # This form is useful when passing to Qiskit Aer
        return (self.gamma, self.lamb)


def _validate_mps(mps: QiskitAerMPS, /) -> None:
    if len(mps.gamma) != len(mps.lamb) + 1:
        raise RuntimeError("The lambda matrix array has an unexpected length.")


def _validate_mps_compatibility(mps1: QiskitAerMPS, mps2: QiskitAerMPS, /) -> None:
    """Ensure that the matrix-product states are defined on the same number of qubits."""
    if len(mps1.gamma) != len(mps2.gamma):
        raise ValueError(
            "The matrix-product states have different numbers of qubits "
            f"({len(mps1.gamma)} vs. {len(mps2.gamma)})."
        )
    _validate_mps(mps1)
    _validate_mps(mps2)


@dispatch
def compute_overlap(mps1: QiskitAerMPS, mps2: QiskitAerMPS, /) -> complex:
    """
    Compute dot product between MPS decompositions of two quantum states:
    ``< mps1 | mps2 >``.

    NOTE: Unlike numpy.dot, this conjugates the first argument.

    Args:
        mps1: MPS decomposition of the left state.
        mps2: MPS decomposition of the right state.

    Returns:
        Complex dot product value.
    """
    num_qubits = len(mps1.gamma)
    _validate_mps_compatibility(mps1, mps2)

    # Remove singleton dimension, so the product "a_b" becomes a matrix.
    # This variable accumulates the result of contraction of two states.
    a = np.squeeze(mps1.gamma[0], axis=1)
    b = np.squeeze(mps2.gamma[0], axis=1)
    a_b = np.tensordot(np.conj(a), b, axes=([0], [0]))

    for n in range(1, num_qubits):
        # Multiply the matrix "a_b" by Diag(Lambda1) and Diag(Lambda2)
        # on respective dimensions.
        a_b *= np.expand_dims(mps1.lamb[n - 1], axis=1)
        a_b *= np.expand_dims(mps2.lamb[n - 1], axis=0)
        # Contraction of the next couple of Gamma tensors.
        a_b = np.tensordot(a_b, np.conj(mps1.gamma[n]), axes=([0], [1]))
        a_b = np.tensordot(a_b, mps2.gamma[n], axes=([0, 1], [1, 0]))

    return complex(a_b.item())


@dispatch
def _compute_overlap_with_local_gate_applied(
    mps1: QiskitAerMPS, gate: Gate, qubit: int, mps2: QiskitAerMPS, /
) -> complex:
    """
    Computes dot product between MPS decompositions of two quantum states
    ``< mps1 | G | mps2 >`` with a single local (1-qubit) gate in the middle.

    NOTE: Unlike numpy.dot, this conjugates the first argument.

    Args:
        mps1: MPS decomposition of the left state.
        gate: Gate to apply.
        qubit: position of a qubit where a local gate is applied.
        mps2: MPS decomposition of the right state.

    Returns:
        complex dot product value.
    """
    num_qubits = len(mps1.gamma)
    _validate_mps_compatibility(mps1, mps2)

    if qubit not in range(num_qubits):
        raise IndexError(f"Invalid qubit index for {num_qubits} qubits: {qubit}")
    if gate.num_qubits != 1:
        raise ValueError("The gate must act on a single qubit.")
    # The following line is expected to always succeed, since plum-dispatch has
    # already identified it as a [unitary] Gate.
    gate_matrix = gate.to_matrix()

    # Remove singleton dimension, so the product "a_b" becomes a matrix.
    # This variable accumulates the result of contraction of two states.
    # Multiply the second Gamma tensor by the gate matrix, if necessary.
    a = np.squeeze(mps1.gamma[0], axis=1)
    b = np.squeeze(mps2.gamma[0], axis=1)
    if qubit == 0:
        a_b = np.tensordot(np.conj(a), gate_matrix @ b, axes=([0], [0]))
    else:
        a_b = np.tensordot(np.conj(a), b, axes=([0], [0]))

    for n in range(1, num_qubits):
        # Multiply the matrix "a_b" by Diag(Lambda1) and Diag(Lambda2)
        # on respective dimensions.
        a_b *= np.expand_dims(mps1.lamb[n - 1], axis=1)
        a_b *= np.expand_dims(mps2.lamb[n - 1], axis=0)
        # Contraction of the next couple of Gamma tensors.
        # Multiply the second Gamma tensor by the gate matrix, if necessary.
        a_b = np.tensordot(a_b, np.conj(mps1.gamma[n]), axes=([0], [1]))
        if qubit == n:
            a_b = np.tensordot(
                a_b,
                np.tensordot(gate_matrix, mps2.gamma[n], axes=([1], [0])),
                axes=([0, 1], [1, 0]),
            )
        else:
            a_b = np.tensordot(a_b, mps2.gamma[n], axes=([0, 1], [1, 0]))

    return complex(a_b.item())
