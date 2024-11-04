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
import pytest
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import (
    generate_time_evolution_circuit,
    generate_xyz_hamiltonian,
)
from scipy.optimize import minimize

from qiskit_addon_aqc_tensor.ansatz_generation import (
    AnsatzBlock,
    generate_ansatz_from_circuit,
)
from qiskit_addon_aqc_tensor.objective import OneMinusFidelity
from qiskit_addon_aqc_tensor.simulation import (
    compute_overlap,
    tensornetwork_from_circuit,
)


@pytest.fixture
def circuit_pair():
    hamiltonian = generate_xyz_hamiltonian(
        CouplingMap([(0, 1), (1, 2), (2, 3)]),
        coupling_constants=(1.2, 0.4, 1.0),
        ext_magnetic_field=(0.0, 0.0, 0.2),
    )
    evolution_time = 0.2
    target_circuit = generate_time_evolution_circuit(
        hamiltonian, synthesis=SuzukiTrotter(reps=3), time=evolution_time
    )
    good_circuit = generate_time_evolution_circuit(
        hamiltonian, synthesis=SuzukiTrotter(reps=1), time=evolution_time
    )
    return good_circuit, target_circuit


def test_basic_workflow(available_backend_fixture, circuit_pair):
    simulator_settings = available_backend_fixture
    good_circuit, target_circuit = circuit_pair
    target_mps = tensornetwork_from_circuit(target_circuit, simulator_settings)
    good_mps = tensornetwork_from_circuit(good_circuit, simulator_settings)
    initial_fidelity = abs(compute_overlap(good_mps, target_mps)) ** 2
    ansatz, initial_parameters = generate_ansatz_from_circuit(
        good_circuit, qubits_initially_zero=True
    )
    objective = OneMinusFidelity(target_mps, ansatz, simulator_settings)
    result = minimize(
        objective,
        initial_parameters,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5},
    )
    assert result.status == 1  # max iterations reached
    final_parameters = result.x
    final_circuit = ansatz.assign_parameters(final_parameters).decompose(AnsatzBlock)
    final_mps = tensornetwork_from_circuit(final_circuit, simulator_settings)
    final_fidelity = abs(compute_overlap(final_mps, target_mps)) ** 2
    assert final_fidelity <= 1
    assert initial_fidelity < final_fidelity
    assert initial_fidelity >= 0
