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
r"""Functions for computing the gradient of :math:`\langle 0 | V^{\dagger}(\vec\theta) | \psi \rangle`.

The purpose of this module is to enable the objective function evaluation (:mod:`qiskit_addon_aqc_tensor.objective`)
to have access to the gradient alongside the function's value.

This module depends on the abstract tensor-network interface provided by :mod:`qiskit_addon_aqc_tensor.simulation`
and is thus agnostic to the tensor-network backend used.

.. currentmodule:: qiskit_addon_aqc_tensor.gradient

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    preprocess_circuit_for_backtracking
    compute_gradient_of_tensornetwork_overlap
"""
from __future__ import annotations

import copy
from functools import partial
from typing import Callable, Sequence

import numpy as np
from qiskit.circuit import (
    Gate,
    Operation,
    Parameter,
    ParameterExpression,
    QuantumCircuit,
)
from qiskit.circuit.library import XGate, YGate, ZGate

from .abstract import (
    TensorNetworkSimulationSettings,
    TensorNetworkState,
    _apply_one_qubit_gate_inplace,
    _apply_two_qubit_gate_inplace,
    _compute_overlap_with_local_gate_applied,
)


def _compute_gate_to_parameter_mapping(qc: QuantumCircuit, /) -> dict[str, int]:
    """Compute mapping between the parameter name and the index of the corresponding entry in the :class:`.ParameterVector` object.

    Recall, Qiskit assigns the parameters sorted by parameter name.
    This remains unique under circuit transformation, and therefore binding
    values at different compilation stages always gives the same result.
    https://quantumcomputing.stackexchange.com/questions/32650/in-what-order-does-qiskit-assign-parameter-values

    Args:
        qc: Quantum circuit.

    Returns:
        Parameter_name-to-index mapping.
    """
    if qc.num_parameters == 0:
        raise ValueError(
            "Expects parametric circuit using ParameterVector object.\n"
            "That is, placeholders are expected rather than concrete\n"
            "values for the variable parameters. The circuit specified\n"
            "has no variable parameters. Check that the function\n"
            "assign_parameters() has not been applied to this circuit."
        )

    pname2index = {str(p.name): i for i, p in enumerate(qc.parameters)}

    if len(pname2index) != qc.num_parameters:  # pragma: no cover
        # This might not actually be a problem (or it might be easy to work
        # around it), but we need to double check.  Also, it may not even be
        # possible to reach this code path through standard qiskit use.
        raise ValueError("It appears some parameter is contained in qc.parameters more than once.")

    return pname2index


# Function that takes two backtracking vectors (in TNS format) and (optionally)
# vectors of parameters and gradients, and updates the backtracking vectors
# in place.
GateAction = Callable[[TensorNetworkState, TensorNetworkState, np.ndarray, np.ndarray], None]


def preprocess_circuit_for_backtracking(
    qc: QuantumCircuit,
    settings: TensorNetworkSimulationSettings,
    /,
) -> list[GateAction]:
    """Preprocess an ansatz circuit for gradient backtracking.

    Returns:
        A list of gate actions in the given circuit to enable backtracking via :func:`compute_gradient_of_tensornetwork_overlap`.
    """
    pname2index = _compute_gate_to_parameter_mapping(qc)

    gate_actions: list[GateAction] = []
    for i, circuit_instruction in enumerate(qc.data):
        operation = circuit_instruction.operation
        qubit_indices = [qc.find_bit(qubit)[0] for qubit in circuit_instruction.qubits]
        try:
            action_generator = _preprocessors[operation.name]
        except KeyError as ex:
            raise ValueError(
                f"Expects a gate from the list of basis ones: "
                f"'{_basis_gates()}', got '{operation.name}' instead."
            ) from ex
        action = action_generator(pname2index, operation, qubit_indices, settings)
        if action is not None:
            gate_actions.append(action)

    return gate_actions


_preprocessors: dict[
    str,
    Callable[
        [dict[str, int], Operation, Sequence[int], TensorNetworkSimulationSettings],
        GateAction | None,
    ],
] = {}


def _register_preprocessor(*args):
    def g(f):
        for name in args:
            assert name not in _preprocessors
            _preprocessors[name] = f
        return f

    return g


def _basis_gates() -> list[str]:
    # Qiskit 2.0 no longer accepts "barrier" as a basis gate, so we remove it
    # from the returned list.
    return list(_preprocessors.keys() - {"barrier"})


@_register_preprocessor("h")
@_register_preprocessor("s")
@_register_preprocessor("sdg")
@_register_preprocessor("x")
@_register_preprocessor("y")
@_register_preprocessor("z")
def _preprocess_one_qubit_parameterless(
    _pname2index: dict[str, int],
    operation: Operation,
    qubit_indices: Sequence[int],
    _: TensorNetworkSimulationSettings,
) -> GateAction | None:
    assert len(qubit_indices) == 1
    return partial(_one_qubit_const, gate=operation, qubit=qubit_indices[0])


_parametrized_rotation_to_pauli = {
    "rx": XGate(),
    "ry": YGate(),
    "rz": ZGate(),
}


@_register_preprocessor("rx")
@_register_preprocessor("ry")
@_register_preprocessor("rz")
def _preprocess_one_qubit_parametrized_pauli_rotation(
    pname2index: dict[str, int],
    operation: Operation,
    qubit_indices: Sequence[int],
    _: TensorNetworkSimulationSettings,
) -> GateAction | None:
    assert len(qubit_indices) == 1
    if not operation.is_parameterized():
        return partial(_one_qubit_const, gate=operation, qubit=qubit_indices[0])
    expr = operation.params[0]
    if len(expr.parameters) != 1:
        raise ValueError("Expression cannot contain more than one Parameter")
    param = next(iter(expr.parameters))
    derivative = expr.gradient(param)
    if isinstance(derivative, ParameterExpression):
        raise ValueError(
            "ParameterExpression's derivative must be a floating-point number, "
            "i.e., the expression must be in the form ax + b."
        )
    # The below line maps RZGate to ZGate, etc.
    pauli = _parametrized_rotation_to_pauli[operation.name]
    return partial(
        _one_qubit_parametrized_pauli_rotation,
        qubit=qubit_indices[0],
        gate=operation,
        pauli=pauli,
        angle_no=pname2index[str(param.name)],
        derivative=derivative,
    )


@_register_preprocessor("cx")
@_register_preprocessor("cz")
@_register_preprocessor("ecr")
def _preprocess_two_qubit_parameterless(
    _pname2index: dict[str, int],
    operation: Operation,
    qubit_indices: Sequence[int],
    settings: TensorNetworkSimulationSettings,
) -> GateAction | None:
    assert len(qubit_indices) == 2
    return partial(
        _two_qubit_const,
        gate=operation,
        q0=qubit_indices[0],
        q1=qubit_indices[1],
        settings=settings,
    )


@_register_preprocessor("barrier")
def _preprocess_no_op(
    _pname2index: dict[str, int],
    _operation: Operation,
    _qubit_indices: Sequence[int],
    _: TensorNetworkSimulationSettings,
) -> GateAction | None:
    return None


def _one_qubit_const(
    w: TensorNetworkState,
    z: TensorNetworkState,
    _: np.ndarray,
    __: np.ndarray,
    /,
    *,
    qubit: int,
    gate: Gate,
) -> None:
    # XXX: Check that this does not assume Hermitian or anything. None of the
    # _const_R[x,y,z] functions performed any complex conjugation here, so they
    # must be right.
    _apply_one_qubit_gate_inplace(w, gate, qubit)
    _apply_one_qubit_gate_inplace(z, gate, qubit)


def _two_qubit_const(
    w: TensorNetworkState,
    z: TensorNetworkState,
    _: np.ndarray,
    __: np.ndarray,
    /,
    *,
    q0: int,
    q1: int,
    gate: Gate,
    settings: TensorNetworkSimulationSettings,
):
    _apply_two_qubit_gate_inplace(w, gate, q0, q1, settings)
    _apply_two_qubit_gate_inplace(z, gate, q0, q1, settings)


def _one_qubit_parametrized_pauli_rotation(
    w: TensorNetworkState,
    z: TensorNetworkState,
    thetas: np.ndarray,
    grad: np.ndarray,
    /,
    *,
    gate: Gate,
    pauli: XGate | YGate | ZGate,
    angle_no: int,
    qubit: int,
    derivative: float,
):
    value = thetas[angle_no]
    expr = gate.params[0]
    if not isinstance(expr, Parameter):
        # Must be an expression
        value = expr.bind({next(iter(expr.parameters)): value}).numeric()
    _apply_one_qubit_gate_inplace(w, type(gate)(value), qubit)
    _apply_one_qubit_gate_inplace(z, type(gate)(value), qubit)
    # 0.5j * <G@w|z> where G is the Pauli corresponding to the single-qubit rotation
    # 0.5j is equal to np.conj(-0.5j)
    grad[angle_no] += (
        0.5j * derivative * _compute_overlap_with_local_gate_applied(w, pauli, qubit, z)
    )


def compute_gradient_of_tensornetwork_overlap(
    *,
    gate_actions: list[GateAction],
    thetas: np.ndarray,
    lhs: TensorNetworkState,
    vdagger_rhs: TensorNetworkState,
) -> np.ndarray:
    r"""Compute gradient of the dot product :math:`\langle \mathrm{lhs} | V^\dagger(\vec\theta) | \mathrm{rhs} \rangle` using tensor networks.

    Here, :math:`V` is a parametric circuit (ansatz), :math:`| \mathrm{rhs} \rangle` is a target state
    (right-hand side vector as a :class:`.TensorNetworkState`), and :math:`| \mathrm{lhs} \rangle` is a low-entangled
    left-hand side vector as a :class:`.TensorNetworkState`, typically equal to :math:`|0\rangle`.

    **Note**: It is assumed that ``vdagger_rhs`` is the state that represents :math:`V^\dagger(\vec\theta) | \mathrm{rhs} \rangle`.
    This same state can be used to calculate the value of the overlap between :math:`V(\vec\theta) | \mathrm{lhs} \rangle` and :math:`| \mathrm{rhs} \rangle`, and by reusing
    it here we save a fair number of computations.

    Args:
        gate_actions: list of pre-processed gate actions (functions) provided by :func:`preprocess_circuit_for_backtracking`.
        thetas: angular parameters of the circuit.
        lhs: left-hand side vector as a :class:`.TensorNetworkState`.
        vdagger_rhs: :math:`V^\dagger(\vec\theta)` times the right-hand side vector as a :class:`.TensorNetworkState`.

    Returns:
        Vector of complex gradients.
    """
    # w, z notation follows the end of Sec. IV of
    # https://arxiv.org/abs/2205.04025v1
    w = copy.deepcopy(lhs)  # deep copy! 1st state for back-tracking
    z = copy.deepcopy(vdagger_rhs)  # 2nd state for back-tracking
    grad = np.zeros(len(thetas), dtype=complex)

    for action in gate_actions:
        action(w, z, thetas, grad)

    return grad


__all__ = [
    "preprocess_circuit_for_backtracking",
    "compute_gradient_of_tensornetwork_overlap",
]
