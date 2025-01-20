# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for backwards compatibility of workflows.

Tests in this file should *not* be updated as API changes are made.  The whole
point of these tests is to make sure old workflows don't intentionally break.
If a workflow is no longer supported, then that test should be removed.

All imports should be in the test functions themselves, to test the stability
of import locations.

It's okay and encouraged to filter deprecation warnings in this file, because
the entire point is to ensure these workflows continue to work, *not* that they
necessarily work without deprecation warnings.

Additionally, all below tests should be excluded from coverage calculations.
"""

import pytest


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skipforcoverage
def test_v0_1_tutorial_workflow():
    """v0.1 tutorial workflow, using ``OneMinusFidelity``

    https://qiskit.github.io/qiskit-addon-aqc-tensor/tutorials/01_initial_state_aqc.html
    """
    from qiskit.transpiler import CouplingMap
    from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian

    # Generate some coupling map to use for this example
    coupling_map = CouplingMap.from_heavy_hex(3, bidirectional=False)

    # Choose a 10-qubit circle on this coupling map
    reduced_coupling_map = coupling_map.reduce([0, 13, 1, 14, 10, 16, 4, 15, 3, 9])

    # Get a qubit operator describing the Ising field model
    hamiltonian = generate_xyz_hamiltonian(
        reduced_coupling_map,
        coupling_constants=(0.0, 0.0, 1.0),
        ext_magnetic_field=(0.4, 0.0, 0.0),
    )

    from qiskit.synthesis import SuzukiTrotter
    from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit

    aqc_evolution_time = 0.5
    aqc_target_num_trotter_steps = 5

    aqc_target_circuit = generate_time_evolution_circuit(
        hamiltonian,
        synthesis=SuzukiTrotter(reps=aqc_target_num_trotter_steps),
        time=aqc_evolution_time,
    )

    subsequent_evolution_time = 0.2
    subsequent_num_trotter_steps = 1

    subsequent_circuit = generate_time_evolution_circuit(
        hamiltonian,
        synthesis=SuzukiTrotter(reps=subsequent_num_trotter_steps),
        time=subsequent_evolution_time,
    )

    aqc_comparison_num_trotter_steps = int(
        subsequent_num_trotter_steps / subsequent_evolution_time * aqc_evolution_time
    )
    aqc_comparison_num_trotter_steps

    comparison_circuit = generate_time_evolution_circuit(
        hamiltonian,
        synthesis=SuzukiTrotter(reps=aqc_comparison_num_trotter_steps),
        time=aqc_evolution_time,
    )

    from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit

    aqc_ansatz_num_trotter_steps = 1

    aqc_good_circuit = generate_time_evolution_circuit(
        hamiltonian,
        synthesis=SuzukiTrotter(reps=aqc_ansatz_num_trotter_steps),
        time=aqc_evolution_time,
    )

    aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(
        aqc_good_circuit, qubits_initially_zero=True
    )

    from qiskit_aer import AerSimulator

    simulator_settings = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=100,
    )

    from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit

    aqc_target_mps = tensornetwork_from_circuit(aqc_target_circuit, simulator_settings)

    from qiskit_addon_aqc_tensor.simulation import compute_overlap

    comparison_mps = tensornetwork_from_circuit(comparison_circuit, simulator_settings)
    comparison_fidelity = abs(compute_overlap(comparison_mps, aqc_target_mps)) ** 2
    comparison_fidelity

    from scipy.optimize import OptimizeResult, minimize

    from qiskit_addon_aqc_tensor.objective import OneMinusFidelity

    objective = OneMinusFidelity(aqc_target_mps, aqc_ansatz, simulator_settings)

    stopping_point = 1 - comparison_fidelity

    def callback(intermediate_result: OptimizeResult):
        print(f"Intermediate result: Fidelity {1 - intermediate_result.fun:.8}")
        if intermediate_result.fun < stopping_point:
            # Good enough for now
            raise StopIteration

    result = minimize(
        objective,
        aqc_initial_parameters,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 100},
        callback=callback,
    )
    if result.status not in (
        0,
        1,
        99,
    ):  # 0 => success; 1 => max iterations reached; 99 => early termination via StopIteration
        raise RuntimeError(f"Optimization failed: {result.message} (status={result.status})")

    print(f"Done after {result.nit} iterations.")
    aqc_final_parameters = result.x

    final_circuit = aqc_ansatz.assign_parameters(aqc_final_parameters)
    final_circuit.compose(subsequent_circuit, inplace=True)
