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

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Utilities for ansatz generation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit.circuit import (
    Gate,
    Parameter,
    ParameterVector,
    QuantumCircuit,
)
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer, TwoQubitWeylDecomposition


class AnsatzBlock(Gate):
    """Ansatz block.

    This is the base class of all blocks returned by
    :func:`generate_ansatz_from_circuit`.
    """

    def __init__(self, params: Sequence[Parameter]):
        """Initialize the ansatz block.

        Args:
            params: Sequence of parameters.
        """
        if len(params) != self.ansatz_num_params:
            raise ValueError("Wrong number of parameters")
        super().__init__(
            self.ansatz_name.lower(),
            self.ansatz_num_qubits,
            params,
            label=self.ansatz_name,
        )


class OneQubitAnsatzBlock(AnsatzBlock):
    """One-qubit ansatz block."""

    ansatz_num_qubits = 1


class TwoQubitAnsatzBlock(AnsatzBlock):
    """Two-qubit ansatz block."""

    ansatz_num_qubits = 2


class ZXZ(OneQubitAnsatzBlock):
    """One-qubit ansatz block based on the ZXZ decomposition.

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :context: reset

       from qiskit.circuit import ParameterVector, QuantumCircuit
       from qiskit_addon_aqc_tensor.ansatz_generation import ZXZ

       qc = QuantumCircuit(1)
       qc.append(ZXZ(ParameterVector("x", 3)), (0,))
       qc.draw("mpl")

    The above ZXZ block is equivalent to the following circuit:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :context: close-figs

       qc.decompose().draw("mpl")
    """

    ansatz_name = "ZXZ"
    ansatz_num_params = 3

    def _define(self) -> None:
        qc = QuantumCircuit(self.ansatz_num_qubits, name=self.ansatz_name)
        qc.rz(self.params[0], 0)
        qc.rx(self.params[1], 0)
        qc.rz(self.params[2], 0)
        self.definition = qc


class KAK(TwoQubitAnsatzBlock):
    """Two-qubit ansatz block based on the KAK decomposition.

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :context: reset

       from qiskit.circuit import ParameterVector, QuantumCircuit
       from qiskit_addon_aqc_tensor.ansatz_generation import KAK

       qc = QuantumCircuit(2)
       qc.append(KAK(ParameterVector("x", 3)), (0, 1))
       qc.draw("mpl")

    The above KAK block is equivalent to the following circuit:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :context: close-figs

       qc.decompose().draw("mpl")
    """

    ansatz_name = "KAK"
    ansatz_num_params = 3

    def _define(self) -> None:
        qc = QuantumCircuit(self.ansatz_num_qubits, name=self.ansatz_name)
        # This implements the two-qubit portion of Fig. 3 in
        # https://arxiv.org/abs/2301.08609v5
        qc.sdg(1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2 - 2 * self.params[2], 0)
        qc.ry(2 * self.params[0] - np.pi / 2, 1)
        qc.cx(0, 1)
        qc.ry(np.pi / 2 - 2 * self.params[1], 1)
        qc.cx(1, 0)
        qc.s(0)
        self.definition = qc


def _allocate_parameters(params: ParameterVector, n: int) -> tuple[list[Parameter], range]:
    m = len(params)
    params.resize(m + n)
    return params[m:], range(m, m + n)


def _nonidle_qubits(qc: QuantumCircuit, /):
    return {
        qubit
        for inst in qc.data
        for qubit in inst.qubits
        if not getattr(inst.operation, "_directive", False)
    }


def generate_ansatz_from_circuit(
    qc: QuantumCircuit,
    /,
    *,
    qubits_initially_zero: bool = False,
    parameter_name: str = "theta",
) -> tuple[QuantumCircuit, list[float]]:
    r"""Generate an ansatz from the two-qubit connectivity structure of a circuit.

    See the `explanatatory material
    <https://qiskit.github.io/qiskit-addon-aqc-tensor/explanation/index.html#ansatz-generation-motivation>`__
    for motivation.

    Args:
        qc: A circuit, which is assumed to be unitary.  Barriers are ignored.
        qubits_initially_zero: If ``True``, the first Z rotation on each qubit
            is fixed to zero because such a rotation has no effect on the state
            :math:`|0\rangle`.
        parameter_name: Name for the :class:`~qiskit.circuit.ParameterVector`
            representing the free parameters in the returned ansatz circuit.

    Returns:
        ``(ansatz, parameter_values)`` such that ``ansatz.assign_parameters(parameter_values)``
        is equivalent to ``qc`` up to a global phase.

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

    If the above circuit is passed to :func:`.generate_ansatz_from_circuit`, it will return an ansatz with parametrized two-qubit KAK rotations in the same locations as the input:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: close-figs

       from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit

       ansatz, initial_params = generate_ansatz_from_circuit(
           qc, qubits_initially_zero=True, parameter_name="x"
       )
       ansatz.draw("mpl")

    Note that in the generated ansatz, all consecutive single-qubit gates are collapsed into the same ZXZ block, and all consecutive two-qubit gates are collapsed into a single KAK block, up to single-qubit rotations.

    Further, the :func:`.generate_ansatz_from_circuit` function provides parameters which, when bound to the ansatz, will result in a circuit equivalent to the original one, up to a global phase:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: close-figs

       ansatz.assign_parameters(initial_params).draw("mpl")

    A 1D Trotter circuit leads to a similar result, with its characteristic brickwork structure:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: reset

       from rustworkx.generators import path_graph
       from qiskit.synthesis import SuzukiTrotter
       from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit, generate_xyz_hamiltonian

       hamiltonian = generate_xyz_hamiltonian(
           path_graph(6),
           coupling_constants=(0.0, 0.0, 1.0),
           ext_magnetic_field=(0.4, 0.0, 0.0),
       )

       good_circuit = generate_time_evolution_circuit(
           hamiltonian,
           synthesis=SuzukiTrotter(reps=2),
           time=1.0,
       )

       good_circuit.draw("mpl", initial_state=True)

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:
       :context: close-figs

       from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit

       ansatz, initial_params = generate_ansatz_from_circuit(
           good_circuit, qubits_initially_zero=True, parameter_name="x"
       )
       ansatz.assign_parameters(initial_params).draw("mpl", initial_state=True)
    """
    # FIXME: handle classical bits, measurements, resets, and barriers.  maybe
    # conditions too?
    num_qubits = qc.num_qubits
    ansatz = QuantumCircuit(*qc.qregs, *qc.cregs)
    param_vec = ParameterVector(parameter_name)
    initial_params: list[float] = []

    decomposer = OneQubitEulerDecomposer("ZXZ")

    partner = [None] * num_qubits
    singles: list[list[Gate] | None] = [None] * num_qubits
    couples: dict[tuple[int, int], QuantumCircuit] = {}
    free_params: dict[int | tuple[int, int], range] = {}

    def set_zxz_params_from_mat(q: int, mat) -> None:
        # Following the variable convention at
        # https://docs.quantum.ibm.com/api/qiskit/qiskit.synthesis.OneQubitEulerDecomposer
        theta, phi, lamb = decomposer.angles(mat)
        fp = free_params[q]
        values: tuple[float, ...] = lamb, theta, phi
        if len(fp) == 2:
            # Must be initial gate, where the Z rotation has been dropped.
            # This makes sense if we assume the input state to this ZXZ block
            # is |0>.
            values = values[1:]
        for j, r in zip(fp, values):
            initial_params[j] = r

    def perform_separation(q0: int, q1: int):
        if q0 > q1:
            q0, q1 = q1, q0
        partner[q0] = None
        partner[q1] = None
        couple_qc = couples[q0, q1]
        mat = Operator(couple_qc).data
        d = TwoQubitWeylDecomposition(mat)
        singles[q0] = [UnitaryGate(d.K1r)]
        singles[q1] = [UnitaryGate(d.K1l)]
        fp01 = free_params[q0, q1]
        initial_params[fp01[0]] = d.a
        initial_params[fp01[1]] = d.b
        initial_params[fp01[2]] = d.c
        set_zxz_params_from_mat(q0, d.K2r)
        set_zxz_params_from_mat(q1, d.K2l)
        del couples[q0, q1]
        free_params[q0] = free_params[q0, q1][3:6]
        free_params[q1] = free_params[q0, q1][6:9]
        del free_params[q0, q1]

    active_qubits = sorted([qc.find_bit(q)[0] for q in _nonidle_qubits(qc)])
    for q in active_qubits:
        params, free_params[q] = _allocate_parameters(param_vec, 2 if qubits_initially_zero else 3)
        initial_params.extend([np.nan] * len(params))
        if qubits_initially_zero:
            params.insert(0, 0.0)
        ansatz.append(ZXZ(params), (q,))
        singles[q] = []

    for inst in qc.data:
        # FIXME: make sure it's unitary in the code paths where we make that assumption
        if inst.operation.name == "barrier":
            pass
        elif len(inst.qubits) == 1:
            if inst.clbits != ():
                raise ValueError("Circuits which operate on classical bits are not yet supported.")
            q = qc.find_bit(inst.qubits[0])[0]
            p = partner[q]
            if p is not None:
                # It's partnered
                qmin = min(q, p)
                qmax = max(q, p)
                couples[qmin, qmax].append(inst.operation, ((0 if q == qmin else 1),))
            else:
                # It's single
                singles[q].append(inst.operation)
        elif len(inst.qubits) == 2:
            q0, q1 = [qc.find_bit(q)[0] for q in inst.qubits]
            swapped = q0 > q1
            if swapped:
                q0, q1 = q1, q0
            if partner[q0] == q1:
                # They are already in a committed relationship with each other
                couple_qc = couples[q0, q1]
            else:
                # We must form a partnership.
                # Start by ensuring everyone is single
                p0 = partner[q0]
                if p0 is not None:
                    perform_separation(q0, p0)
                p1 = partner[q1]
                if p1 is not None:
                    perform_separation(q1, p1)
                # Form the union
                couple_qc = QuantumCircuit(2)
                couples[q0, q1] = couple_qc
                for op in singles[q0]:
                    couple_qc.append(op, (0,))
                for op in singles[q1]:
                    couple_qc.append(op, (1,))
                singles[q0] = None
                singles[q1] = None
                partner[q0] = q1
                partner[q1] = q0
                # Update the ansatz
                params, free_params[q0, q1] = _allocate_parameters(param_vec, 9)
                initial_params.extend([np.nan] * 9)
                ansatz.append(KAK(params[0:3]), (q0, q1))
                ansatz.append(ZXZ(params[3:6]), (q0,))
                ansatz.append(ZXZ(params[6:9]), (q1,))
            couple_qc.append(inst.operation, (1, 0) if swapped else (0, 1))
        else:
            raise ValueError(
                "Only one- and two-qubit operations are allowed in the original circuit."
            )

    while couples:
        (q0, q1), couple_qc = next(
            iter(couples.items())
        )  # Must do this because we can't modify a dict while iterating it
        perform_separation(q0, q1)
    for q, ops in enumerate(singles):
        if ops is None:
            # Must be an idle qubit
            continue
        single_qc = QuantumCircuit(1)
        for op in ops:
            single_qc.append(op, (0,))
        mat = Operator(single_qc).data
        set_zxz_params_from_mat(q, mat)
        del free_params[q]

    assert not free_params

    return ansatz, initial_params


# Reminder: update the RST file in docs/apidocs when adding new interfaces.
__all__ = [
    "generate_ansatz_from_circuit",
    "AnsatzBlock",
    "OneQubitAnsatzBlock",
    "TwoQubitAnsatzBlock",
    "ZXZ",
    "KAK",
]
