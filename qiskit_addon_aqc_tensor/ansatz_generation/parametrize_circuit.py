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

from .from_connectivity import _allocate_parameters


def parametrize_circuit(
    qc: QuantumCircuit,
    /,
    *,
    parameter_name: str = "theta",
) -> tuple[QuantumCircuit, list[float]]:
    r"""Create a parametrized version of a circuit.

    Given a quantum circuit, constructs another quantum circuit which is identical
    except that any gates with numerical parameters are replaced by gates (of the same
    type) with free parameters. The new circuit is returned along with a list containing
    the original values of the parameters.

    Args:
        qc: The quantum circuit to parametrize.
        parameter_name: Name for the :class:`~qiskit.circuit.ParameterVector`
            representing the free parameters in the returned ansatz circuit.

    Returns:
        ``(ansatz, parameter_values)`` such that ``ansatz.assign_parameters(parameter_values)``
        is identical to ``qc``.

    Example:
    ========

    Consider the following circuit as an example:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: reset

       from qiskit import QuantumCircuit

       qc = QuantumCircuit(6)
       qc.rx(0.4, 0)
       qc.ryy(0.2, 2, 3)
       qc.h(2)
       qc.rz(0.1, 2)
       qc.rxx(0.3, 0, 1)
       qc.rzz(0.3, 0, 1)
       qc.cx(2, 1)
       qc.s(1)
       qc.h(4)
       qc.draw("mpl")

    If the above circuit is passed to :func:`.parametrize_circuit`, it will return an ansatz
    obtained from this circuit by replacing numerical parameters with free parameters:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: close-figs

       from qiskit_addon_aqc_tensor import parametrize_circuit

       ansatz, initial_params = parametrize_circuit(qc)
       ansatz.draw("mpl")

    Further, the :func:`.parametrize_circuit` function provides parameters which, when bound to the ansatz, will result in a circuit identical to the original one:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: close-figs

       ansatz.assign_parameters(initial_params).draw("mpl")
    """
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

    for i, param in enumerate(ansatz.parameters):
        if param in qc.parameters:
            initial_params.insert(i, None)

    return ansatz, initial_params
