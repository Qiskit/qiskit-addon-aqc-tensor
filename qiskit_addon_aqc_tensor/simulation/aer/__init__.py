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

"""Qiskit Aer MPS simulator as a tensor network backend."""

from .gradient import _preprocess_for_gradient
from .simulation import (
    QiskitAerSimulationSettings,
    apply_circuit_to_state,
    is_aer_available,
    tensornetwork_from_circuit,
)
from .state import QiskitAerMPS, compute_overlap

# We want the plum-dispatch methods from gradient.py to be available, so we
# have made an unused import from that file.  But now we must reference that
# import so the linter does not complain.
_ = _preprocess_for_gradient

__all__ = [
    "is_aer_available",
    "QiskitAerMPS",
    "QiskitAerSimulationSettings",
    # plum-dispatch methods
    "compute_overlap",
    "apply_circuit_to_state",
    "tensornetwork_from_circuit",
]
