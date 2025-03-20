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

"""Function for constructing a parameterized version of a circuit."""

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit

from .ansatz_generation import _allocate_parameters


def parametrize_circuit(
    qc: QuantumCircuit,
    /,
    *,
    parameter_name: str = "theta",
) -> QuantumCircuit:
    """Create a parametrized version of a circuit."""
    ansatz = QuantumCircuit(*qc.qregs, *qc.cregs)
    param_vec = ParameterVector(parameter_name)
    initial_params: list[float] = []

    for inst in qc.data:
        operation = inst.operation
        original_params = operation.params
        fixed_indices = [
            i for i, val in enumerate(original_params) if not isinstance(val, Parameter)
        ]
        if fixed_indices:
            # Replace all non-Parameter entries with parameters
            operation = operation.copy()
            params = operation.params
            allocated_params, _ = _allocate_parameters(param_vec, len(fixed_indices))
            for i, param in zip(fixed_indices, allocated_params):
                params[i] = param
                initial_params.append(original_params[i])
        ansatz.append(operation, inst.qubits, inst.clbits)

    return ansatz, initial_params
