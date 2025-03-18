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

"""Quimb as a tensor network backend."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

import numpy as np
from plum import ModuleType, clear_all_cache, dispatch
from qiskit.circuit import Gate, Parameter, ParameterExpression, QuantumCircuit
from wrapt import register_post_import_hook

from ...ansatz_generation import AnsatzBlock
from ...objective import OneMinusFidelity
from ..abstract import TensorNetworkSimulationSettings
from ..explicit_gradient import (
    compute_gradient_of_tensornetwork_overlap,
    preprocess_circuit_for_backtracking,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    import quimb.tensor
    from quimb.tensor import Circuit, CircuitMPS
else:
    Circuit = ModuleType("quimb.tensor", "Circuit")
    CircuitMPS = ModuleType("quimb.tensor", "CircuitMPS")


def _on_quimb_import(_module) -> None:
    """Clear plum cache upon loading of :mod:`quimb.tensor`."""
    logger.info("Clearing plum cache after loading of quimb.tensor")
    # https://beartype.github.io/plum/types.html#moduletype
    clear_all_cache()


register_post_import_hook(_on_quimb_import, "quimb.tensor")


def is_quimb_available() -> bool:
    """Return ``True`` is qiskit-quimb is installed, ``False`` otherwise."""
    try:
        metadata_version("qiskit-quimb")
    except PackageNotFoundError:  # pragma: no cover
        return False
    return True


class QuimbCircuitFactory(Protocol):
    """Quimb circuit factory."""

    def __call__(  # noqa: D102
        self, *, N: int, psi0: quimb.tensor.TensorNetworkGenVector | None = None
    ) -> quimb.tensor.Circuit:  # pragma: no cover
        ...


@dataclass
class QuimbSimulator(TensorNetworkSimulationSettings):
    """Settings for Quimb simulator.

    This is compatible with both `Quimb's MPS simulator
    <https://quimb.readthedocs.io/en/latest/tensor-circuit-mps.html>`__,
    which eagerly contracts gates, as well as `Quimb's standard method for
    circuit simulation
    <https://quimb.readthedocs.io/en/latest/tensor-circuit.html>`__.
    """

    #: Callable for constructing the Quimb circuit, e.g., :func:`~quimb.tensor.Circuit` or :func:`~quimb.tensor.CircuitMPS`.
    quimb_circuit_factory: QuimbCircuitFactory
    # Automatic differentiation backend for evaluating gradient.  Options: 'jax', 'autograd', 'torch', etc., or 'explicit' for the original AQC-Tensor gradient implementation in the case of a :func:`~quimb.tensor.CircuitMPS`.
    autodiff_backend: Optional[str] = None
    #: Whether to display a progress bar while applying gates.
    progbar: bool = False

    def _construct_circuit(
        self, qc: QuantumCircuit, /, *, out_state: np.ndarray | None = None, **kwargs
    ):
        import qiskit_quimb

        qc = qc.decompose(AnsatzBlock)
        quimb_circuit_factory = self.quimb_circuit_factory
        circ = quimb_circuit_factory(N=qc.num_qubits, **kwargs)
        gates = qiskit_quimb.quimb_gates(qc)
        circ.apply_gates(gates, progbar=self.progbar)
        if out_state is not None:
            out_state[:] = np.squeeze(circ.psi.to_dense())
        return circ


@dispatch
def compute_overlap(circ1: Circuit, circ2: Circuit, /) -> complex:
    return complex(circ1.psi.H @ circ2.psi)


@dispatch
def tensornetwork_from_circuit(
    qc: QuantumCircuit,
    settings: QuimbSimulator,
    /,
    *,
    out_state: Optional[np.ndarray] = None,
) -> quimb.tensor.Circuit:
    return settings._construct_circuit(qc, out_state=out_state)


@dispatch
def _compute_overlap_with_local_gate_applied(
    circ1: CircuitMPS,
    gate: Gate,
    qubit: int,
    circ2: CircuitMPS,
    /,
) -> complex:
    gate_matrix = gate.to_matrix()
    g_psi_2 = circ2.psi.gate(gate_matrix, qubit)
    return complex(circ1.psi.H @ g_psi_2)


@dispatch
def _apply_one_qubit_gate_inplace(circ: CircuitMPS, gate: Gate, qubit: int, /) -> None:
    """Apply one-qubit gate in place."""
    from qiskit_quimb import quimb_gate

    quimb_gate_ = quimb_gate(gate, (qubit,))
    if quimb_gate_ is not None:
        circ.apply_gate(quimb_gate_)


@dispatch
def _apply_two_qubit_gate_inplace(
    circ: CircuitMPS,
    gate: Gate,
    q0: int,
    q1: int,
    _: QuimbSimulator,
    /,
) -> None:
    """Apply two-qubit gate in place."""
    from qiskit_quimb import quimb_gate

    quimb_gate_ = quimb_gate(gate, (q0, q1))
    if quimb_gate_ is not None:
        circ.apply_gate(quimb_gate_)


@dispatch
def apply_circuit_to_state(
    qc: QuantumCircuit,
    circ0: Circuit,
    settings: QuimbSimulator,
    /,
    *,
    out_state: Optional[np.ndarray] = None,
) -> quimb.tensor.Circuit:
    """Apply a quantum circuit to a tensor network state.

    The input state (``psi``) is not modified.

    Returns:
        The new state.
    """
    return settings._construct_circuit(qc, out_state=out_state, psi0=circ0.psi)


class QiskitQuimbConversionContext:
    """Contains information about Qiskit-to-Quimb conversion, necessary to recover Qiskit parameters."""

    def __init__(self, mapping: list[tuple[int, float, float]], /):
        """Initialize.  Should not be called by users."""
        self._mapping = mapping


def qiskit_ansatz_to_quimb(
    qc: QuantumCircuit, initial_parameters: Sequence[float], /
) -> tuple[quimb.tensor.Circuit, QiskitQuimbConversionContext]:
    """Convert a Qiskit ansatz to a Quimb parametrized circuit."""
    import quimb.tensor as qtn
    from qiskit_quimb import quimb_gate

    qc = qc.decompose(AnsatzBlock)
    if len(initial_parameters) != qc.num_parameters:
        raise ValueError(
            f"{len(initial_parameters)} parameter(s) were passed, but "
            f"the circuit has {qc.num_parameters} parameter(s)."
        )
    circ = qtn.Circuit(qc.num_qubits)
    mapping: list[tuple[int, float, float]] = [(-1, 0.0, 0.0)] * qc.num_parameters
    j = 0
    parameter_lookup: dict[Parameter, int] = {
        param: index for index, param in enumerate(qc.parameters)
    }
    for instruction in qc.data:
        op = instruction.operation
        qubits = [qc.find_bit(qubit)[0] for qubit in instruction.qubits]
        if any(isinstance(p, ParameterExpression) for p in op.params):
            # The current instruction should become a quimb parametrized gate.
            # First, a sanity check.
            if len(op.params) != 1:
                raise ValueError(
                    "This code is not designed to support parametrized gates "
                    "with multiple parameters."
                )
            expr = op.params[0]
            # Extract the parameter
            if len(expr.parameters) != 1:
                raise ValueError("Expression cannot contain more than one Parameter")
            param = next(iter(expr.parameters))
            # Back out the expression.  Make sure it is linear; otherwise we
            # don't know how to invert it, and we need to do this later when
            # converting back to Qiskit parameters.
            m = expr.gradient(param)
            if isinstance(m, ParameterExpression):
                raise ValueError(
                    "The Quimb backend currently requires that each ParameterExpression "
                    f"must be in the form mx + b (not {expr}).  Otherwise, the backend is unable "
                    "to recover the parameter."
                )
            b = expr.bind({param: 0}).numeric()
            # Create an equivalent operation that is not parametrized
            fixed_op = deepcopy(op)
            try:
                index = parameter_lookup[param]
            except KeyError as ex:  # pragma: no cover
                raise RuntimeError(
                    "Unexpected error: Parameter of operation is not listed "
                    "among the circuit's parameters."
                ) from ex
            if mapping[index][0] != -1:
                raise ValueError(
                    "Parameter cannot be repeated in circuit, else "
                    "quimb will attempt to optimize each instance separately."
                )
            mapping[index] = (j, m, b)
            j = j + 1
            fixed_op.params[0] = expr.bind({param: initial_parameters[index]}).numeric()
            # Convert to a quimb gate
            fixed_quimb_gate = quimb_gate(fixed_op, qubits, parametrize=True)
            # Append it to the quimb circuit
            circ.apply_gate(fixed_quimb_gate, tags=["param"], contract=False)
        elif all(np.issubdtype(type(p), np.number) for p in op.params):
            # We can apply the gate as usual, as all parameters are ordinary numbers
            quimb_gate_ = quimb_gate(op, qubits)
            if quimb_gate_ is not None:
                circ.apply_gate(quimb_gate_)
        else:  # pragma: no cover
            raise ValueError("A parameter in the circuit has an unexpected type.")
    for j, _, _ in mapping:
        if j == -1:  # pragma: no cover
            # NOTE: There seems to be no obvious way to trigger this error.
            # Even the following snippet results in the parameter being removed
            # from the circuit.
            #
            # qc = QuantumCircuit(1)
            # x = Parameter("x")
            # qc.rx(x, 0)
            # qc.data[0] = CircuitInstruction(RXGate(np.pi / 3), qubits=[0])
            raise ValueError(
                "Some parameter(s) in the given Qiskit circuit remain unused. "
                "This use case is not currently supported by the Quimb conversion code."
            )
    return circ, QiskitQuimbConversionContext(mapping)


def recover_parameters_from_quimb(
    circ_opt: quimb.tensor.Circuit, ctx: QiskitQuimbConversionContext, /
) -> list[float]:
    """Recover Qiskit circuit parameters from a Quimb circuit."""
    quimb_parametrized_gates = [gate for gate in circ_opt.gates if gate.parametrize]
    mapping = ctx._mapping
    if len(quimb_parametrized_gates) != len(mapping):
        raise ValueError(
            "The length of the mapping in the provided QiskitQuimbConversionContext "
            "does not match the number of parametrized gates in the circuit "
            f"({len(mapping)} vs. {len(quimb_parametrized_gates)})."
        )
    # `(y - b) / m` is the inversion of the parameter expression, which we
    # assumed above to be in the form mx + b.
    return [(float(quimb_parametrized_gates[j].params[0]) - b) / m for (j, m, b) in mapping]


@dispatch
def _preprocess_for_gradient(objective, settings: QuimbSimulator):
    if settings.autodiff_backend is None:
        raise ValueError(
            "Gradient method unspecified. Please specify an autodiff_backend "
            "for the QuimbSimulator object."
        )
    if objective._ansatz is not None:
        ansatz_num_qubits = objective._ansatz.num_qubits
        target_num_qubits = objective._target_tensornetwork.N
        if ansatz_num_qubits != target_num_qubits:
            raise ValueError(
                "Ansatz and target have different numbers of qubits "
                f"({ansatz_num_qubits} vs. {target_num_qubits})."
            )
    if settings.autodiff_backend == "explicit":
        # FIXME: error if target and/or settings could result in non-MPS, in
        # order to prevent a later MethodError from plum
        return _ExplicitGradientContext(objective, settings)
    return _QuimbGradientContext(objective, settings)


class _ExplicitGradientContext:
    def __init__(self, objective, settings):
        self.gate_actions = preprocess_circuit_for_backtracking(objective._ansatz, settings)
        self.lhs_tensornetwork = tensornetwork_from_circuit(
            QuantumCircuit(objective._ansatz.num_qubits), settings
        )  # |0>


@dispatch
def _compute_objective_and_gradient(
    objective: OneMinusFidelity,
    settings: QuimbSimulator,
    preprocess_info: _ExplicitGradientContext,
    x: np.ndarray,
) -> tuple[float, np.ndarray]:
    lhs_tensornetwork = preprocess_info.lhs_tensornetwork
    bound_qc = objective._ansatz.assign_parameters(x)
    vdagger_target = apply_circuit_to_state(
        bound_qc.inverse(), objective._target_tensornetwork, settings
    )
    overlap = compute_overlap(lhs_tensornetwork, vdagger_target)
    objective_value = 1 - abs(overlap) ** 2
    overlap_gradient = compute_gradient_of_tensornetwork_overlap(
        gate_actions=preprocess_info.gate_actions,
        thetas=x,
        lhs=lhs_tensornetwork,
        vdagger_rhs=vdagger_target,
    )
    gradient = -2 * np.real(np.conj(overlap) * overlap_gradient)
    return objective_value, gradient


class _QuimbGradientContext:
    def __init__(self, objective, settings):
        import quimb.tensor as qtn

        self.quimb_ansatz, self.conversion_ctx = qiskit_ansatz_to_quimb(
            objective._ansatz, [0.0] * objective._ansatz.num_parameters
        )
        self.tnopt = qtn.TNOptimizer(
            self.quimb_ansatz,
            **tnoptimizer_objective_kwargs(objective),
            autodiff_backend=settings.autodiff_backend,
        )


@dispatch
def _compute_objective_and_gradient(
    _: OneMinusFidelity,
    __: QuimbSimulator,
    preprocess_info: _QuimbGradientContext,
    qiskit_parameter_values: np.ndarray,
) -> tuple[float, np.ndarray]:
    mapping = preprocess_info.conversion_ctx._mapping

    # Convert parameters qiskit -> quimb (evaluate parameter expressions)
    quimb_parameter_values = np.zeros(len(mapping))
    for i, (j, m, b) in enumerate(mapping):
        quimb_parameter_values[j] = m * qiskit_parameter_values[i] + b

    # Evaluate objective value and gradient using quimb
    val, quimb_gradient = preprocess_info.tnopt.vectorized_value_and_grad(quimb_parameter_values)

    # Convert gradient quimb -> qiskit (divide by derivative of parameter expressions)
    qiskit_gradient = np.zeros(len(mapping))
    for i, (j, m, _) in enumerate(mapping):
        qiskit_gradient[i] = m * quimb_gradient[j]

    return val, qiskit_gradient


@dispatch
def tnoptimizer_objective_kwargs(objective: OneMinusFidelity, /) -> dict[str, Any]:
    """Return keyword arguments for use with :func:`~quimb.tensor.TNOptimizer`.

    - ``loss_fn``
    - ``loss_kwargs``
    """
    import quimb.tensor as qtn

    target = objective.target
    if isinstance(target, qtn.Circuit):
        target = target.psi
    return {
        "loss_fn": oneminusfidelity_loss_fn,
        "loss_kwargs": {"target": target},
    }


def oneminusfidelity_loss_fn(
    circ: quimb.tensor.Circuit, /, *, target: quimb.tensor.TensorNetworkGenVector
):
    """Loss function for use with Quimb, compatible with automatic differentiation.

    See the `introduction to optimization with quimb
    <https://quimb.readthedocs.io/en/latest/tensor-optimization.html>`__
    for details on using :func:`~quimb.tensor.optimize.TNOptimizer`.
    """
    import autoray as ar

    overlap = target.H @ circ.psi
    # we use `autoray.do` to allow arbitrary autodiff backends
    fidelity = ar.do("abs", overlap) ** 2
    return 1 - fidelity


# Reminder: update the RST file in docs/apidocs when adding new interfaces.
__all__ = [
    "is_quimb_available",
    "QuimbCircuitFactory",
    "QuimbSimulator",
    "QiskitQuimbConversionContext",
    "qiskit_ansatz_to_quimb",
    "recover_parameters_from_quimb",
    "tnoptimizer_objective_kwargs",
    # plum-dispatch methods
    "compute_overlap",
    "apply_circuit_to_state",
    "tensornetwork_from_circuit",
]
