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

"""Approximate Quantum Compiling using Tensor Networks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .ansatz_generation import generate_ansatz_from_circuit

try:
    __version__ = version("qiskit-addon-aqc-tensor")
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed
    pass

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
__all__ = [
    "generate_ansatz_from_circuit",
]
