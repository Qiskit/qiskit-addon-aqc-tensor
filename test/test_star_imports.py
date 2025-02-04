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

# Star imports must be at the top-level of the module, so we do them all here
# and then have a single test return success.  These lines ensure that the
# `__all__` list in each module refers to something that actually exists
# and is imported by the module.
from qiskit_addon_aqc_tensor import *
from qiskit_addon_aqc_tensor.ansatz_generation import *
from qiskit_addon_aqc_tensor.objective import *
from qiskit_addon_aqc_tensor.simulation import *
from qiskit_addon_aqc_tensor.simulation.aer import *
from qiskit_addon_aqc_tensor.simulation.explicit_gradient import *
from qiskit_addon_aqc_tensor.simulation.quimb import *


def test_star_imports():
    assert generate_ansatz_from_circuit.__name__ == "generate_ansatz_from_circuit"
    assert KAK.__name__ == "KAK"
    assert preprocess_circuit_for_backtracking.__name__ == "preprocess_circuit_for_backtracking"
    assert MaximizeStateFidelity.__name__ == "MaximizeStateFidelity"
    assert TensorNetworkState.__name__ == "TensorNetworkState"
    assert QiskitAerMPS.__name__ == "QiskitAerMPS"
    assert QuimbSimulator.__name__ == "QuimbSimulator"
