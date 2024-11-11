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

"""Aer MPS simulator configuration and functions that require a simulator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
from plum import ModuleType, clear_all_cache, dispatch
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.result import Result
from wrapt import register_post_import_hook

from ..abstract import TensorNetworkSimulationSettings
from .state import QiskitAerMPS

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from qiskit_aer import AerSimulator
else:
    AerSimulator = ModuleType("qiskit_aer", "AerSimulator")


def _on_aer_import(_module) -> None:
    """Clear plum cache upon loading of :mod:`qiskit_aer`."""
    logger.info("Clearing plum cache after loading of qiskit_aer")
    # https://beartype.github.io/plum/types.html#moduletype
    clear_all_cache()


register_post_import_hook(_on_aer_import, "qiskit_aer")


def is_aer_available() -> bool:
    """Return ``True`` is qiskit-aer is installed, ``False`` otherwise."""
    try:
        metadata_version("qiskit-aer")
    except PackageNotFoundError:
        return False
    return True


@dataclass
class QiskitAerSimulationSettings(TensorNetworkSimulationSettings):
    """Qiskit Aer simulator settings.

    Example usage:

    .. code-block:: python

       from qiskit_aer import AerSimulator
       from qiskit_addon_aqc_tensor.simulation.aer import QiskitAerSimulationSettings

       simulator = AerSimulator(
           method="matrix_product_state",
           matrix_product_state_max_bond_dimension=200,
           matrix_product_state_truncation_threshold=1e-6,
           mps_log_data=True,
       )

       def my_simulation_callback(circuit, result):
           print(f"Simulated circuit of depth {circuit.depth()}")
           metadata = result.results[0].metadata
           print(metadata["MPS_log_data"])

       settings = QiskitAerSimulationSettings(simulator, my_simulation_callback)

    See additional options for when using the ``matrix_product_state`` simulation method
    in the documentation for :class:`~qiskit_aer.AerSimulator`.
    """

    #: Aer simulator, must be configured with method='matrix_product_state'.  Stores configuration like ``max_bond_dimension``, ``truncation_threshold``, etc.
    simulator: AerSimulator

    #: This callable, if provided, is called with ``(circuit, result)`` as arguments immediately after each MPS simulation.
    callback: Callable[[QuantumCircuit, Result], None] | None = None


def _aer_mps_from_circuit(
    qc: QuantumCircuit,
    settings: QiskitAerSimulationSettings | AerSimulator,
    /,
    *,
    out_state: np.ndarray | None = None,
) -> QiskitAerMPS:
    r"""Compute the result of action ``output = circuit @ |0>`` in MPS format.

    Args:
        qc: quantum circuit acting on state :math:`|0\rangle`.
        settings: instance of :class:`.QiskitAerSimulationSettings` or
                  :class:`~qiskit_aer.AerSimulator`.  Either way, the simulator
                  must be configured with ``method="matrix_product_state"``.
        out_state: output array for storing state as a normal vector; *note*,
                   state generation can be a slow and even intractable operation
                   for the large number of qubits; useful for testing only.

    Returns:
        MPS state representation.
    """
    from qiskit_aer import AerSimulator

    if not isinstance(settings, QiskitAerSimulationSettings):
        # Presumably we've been passed an AerSimulator, instead.
        settings = QiskitAerSimulationSettings(settings)
    simulator = settings.simulator

    # Validate inputs
    if not isinstance(simulator, AerSimulator):
        raise TypeError("simulator must be of type AerSimulator.")
    if simulator.options.method != "matrix_product_state":
        raise ValueError("AerSimulator must be configured to use 'matrix_product_state' method.")

    if out_state is not None:
        if not isinstance(out_state, np.ndarray):
            raise TypeError("If provided, `out_state` must be of type numpy.ndarray.")
        if out_state.size != 2**qc.num_qubits:
            raise ValueError(
                f"If provided, `out_state` must have size 2**num_qubits ({2**qc.num_qubits}), "
                f"not {out_state.size}."
            )

    # Copy the input circuit before appending save operation(s)
    qc = qc.copy()
    if out_state is not None:
        qc.save_statevector(label="my_sv")
    qc.save_matrix_product_state(label="my_mps")

    # Run the simulation
    result = simulator.run(qc, shots=1).result()
    data = result.data(0)

    if settings.callback is not None:
        settings.callback(qc, result)

    if out_state is not None:
        np.copyto(out_state, np.asarray(data["my_sv"]))

    return QiskitAerMPS(*data["my_mps"])


@dispatch
def tensornetwork_from_circuit(
    qc: QuantumCircuit,
    settings: Union[QiskitAerSimulationSettings, AerSimulator],
    /,
    *,
    out_state: Optional[np.ndarray] = None,
) -> QiskitAerMPS:
    return _aer_mps_from_circuit(qc, settings, out_state=out_state)


@dispatch
def _apply_one_qubit_gate_inplace(psi: QiskitAerMPS, gate: Gate, qubit: int, /) -> None:
    from qiskit_aer import AerSimulator

    num_qubits = len(psi.gamma)
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(psi._as_tuple())
    qc.append(gate, (qubit,))
    # Parameters such as the bond dimension don't matter for single-qubit
    # operations because there is no need to perform MPS compression.  Hence,
    # we use a simulator with the default settings here.  FIXME: in the future,
    # do this rotation using numpy, without even calling into Aer.
    new_psi = _aer_mps_from_circuit(qc, AerSimulator(method="matrix_product_state"))
    # Replace existing instance
    psi.gamma = new_psi.gamma
    psi.lamb = new_psi.lamb


@dispatch
def _apply_two_qubit_gate_inplace(
    psi: QiskitAerMPS,
    gate: Gate,
    q0: int,
    q1: int,
    settings: Union[QiskitAerSimulationSettings, AerSimulator],
    /,
    *,
    out_state: Optional[np.ndarray] = None,
) -> None:
    num_qubits = len(psi.gamma)
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(psi._as_tuple())
    qc.append(gate, (q0, q1))
    new_psi = _aer_mps_from_circuit(qc, settings, out_state=out_state)
    # Replace existing instance
    psi.gamma = new_psi.gamma
    psi.lamb = new_psi.lamb


@dispatch
def apply_circuit_to_state(
    qc: QuantumCircuit,
    psi: QiskitAerMPS,
    settings: Union[QiskitAerSimulationSettings, AerSimulator],
    /,
    *,
    out_state: Optional[np.ndarray] = None,
) -> QiskitAerMPS:
    # Note, the order of operations is crucial. We compose the circuit after (!)
    # invocation of set_matrix_product_state().
    qc2 = QuantumCircuit(qc.num_qubits)
    qc2.set_matrix_product_state(psi._as_tuple())
    qc2.compose(qc, inplace=True)
    return _aer_mps_from_circuit(qc2, settings, out_state=out_state)
