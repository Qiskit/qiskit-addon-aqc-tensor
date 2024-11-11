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

"""This module provides an interface to tensor-network simulation that is agnostic to which backend is used.

Currently, there is a single implementation, which uses the Qiskit Aer matrix-product state (MPS) simulator as its backend.

Abstract classes
----------------

.. currentmodule:: qiskit_addon_aqc_tensor.simulation

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    TensorNetworkState
    TensorNetworkSimulationSettings

Aer backend (:mod:`qiskit_addon_aqc_tensor.simulation.aer`)

.. currentmodule:: qiskit_addon_aqc_tensor.simulation.aer

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    QiskitAerMPS
    QiskitAerSimulationSettings

Methods.

The following abstract methods provide a common interface that can be used with any supported backend:

.. currentmodule:: qiskit_addon_aqc_tensor.simulation

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    tensornetwork_from_circuit
    apply_circuit_to_state
    compute_overlap
"""

# We must import each tensor-network backend we wish to use so that their
# methods will be available.  But then we must reference each such symbol so
# the linter does not force us to add them to __all__.
from . import aer, quimb
from .abstract import (
    TensorNetworkSimulationSettings,
    TensorNetworkState,
    apply_circuit_to_state,
    compute_overlap,
    tensornetwork_from_circuit,
)

_ = aer
__ = quimb

__all__ = [
    "TensorNetworkState",
    "TensorNetworkSimulationSettings",
    "compute_overlap",
    "apply_circuit_to_state",
    "tensornetwork_from_circuit",
]
