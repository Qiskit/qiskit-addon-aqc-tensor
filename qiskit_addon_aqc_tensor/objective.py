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

"""Code for building and evaluating objective functions used for AQC parameter optimization.

Currently, this module provides the simplest possible objective function, :class:`.MaximizeStateFidelity`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.utils.deprecation import deprecate_func

if TYPE_CHECKING:  # pragma: no cover
    from .simulation.abstract import (
        TensorNetworkSimulationSettings,
        TensorNetworkState,
    )


class MaximizeStateFidelity:
    r"""Simplest possible objective function for use with AQC-Tensor.

    Its definition is given by Eq. (7) in `arXiv:2301.08609v6 <https://arxiv.org/abs/2301.08609v6>`__:

    .. math::
       C = 1 - \left| \langle 0 | V^{\dagger}(\vec\theta) | \psi_\mathrm{target} \rangle \right|^2 .

    Minimizing this function is equivalent to maximizing the pure-state fidelity
    between the state prepared by the ansatz circuit at the current parameter
    point, :math:`V(\vec\theta) |0\rangle`, and the target state,
    :math:`| \psi_\mathrm{target} \rangle`.

    When called with an :class:`~numpy.ndarray` of parameters, this object will return
    ``(objective_value, gradient)`` as a ``tuple[float, numpy.ndarray]``.
    """

    def __init__(
        self,
        target: TensorNetworkState,
        ansatz: QuantumCircuit,
        settings: TensorNetworkSimulationSettings,
    ):
        """Initialize the objective function.

        Args:
            ansatz: Parametrized ansatz circuit.
            target: Target state in tensor-network representation.
            settings: Tensor network simulation settings.
        """
        if ansatz is not None:
            from .ansatz_generation import AnsatzBlock

            ansatz = ansatz.decompose(AnsatzBlock)
        self._ansatz = ansatz
        self._simulation_settings = settings
        self._target_tensornetwork = target
        if settings is not None:
            from .simulation.abstract import _preprocess_for_gradient

            self._preprocessed = _preprocess_for_gradient(self, settings)

    @deprecate_func(
        since="0.2.0",
        package_name="qiskit-addon-aqc-tensor",
        removal_timeline="no earlier than v0.4.0",
        additional_msg=(
            "Going forward, the ``loss_function`` method should be called "
            "instead of calling the instance directly."
        ),
    )
    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:  # pragma: no cover
        """Evaluate ``(objective_value, gradient)`` of function at point ``x``.

        This method is DEPRECATED since v0.2.  The
        :meth:`~.MaximizeStateFidelity.loss_function` method should be called instead.
        """
        return self.loss_function(x)

    def loss_function(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate ``(objective_value, gradient)`` of function at point ``x``."""
        from .simulation.abstract import _compute_objective_and_gradient

        return _compute_objective_and_gradient(
            self, self._simulation_settings, self._preprocessed, x
        )

    @property
    def target(self) -> TensorNetworkState:
        """Target tensor network."""
        return self._target_tensornetwork

    @property
    def ansatz(self) -> QuantumCircuit | None:
        """Parametrized ansatz circuit."""
        return self._ansatz


class OneMinusFidelity(MaximizeStateFidelity):
    """DEPRECATED objective function, equivalent to :class:`.MaximizeStateFidelity`."""

    @deprecate_func(
        since="0.2.0",
        package_name="qiskit-addon-aqc-tensor",
        removal_timeline="no earlier than v0.4.0",
        additional_msg="This class has been renamed to ``MaximizeStateFidelity``.",
    )
    def __init__(
        self,
        target: TensorNetworkState,
        ansatz: QuantumCircuit,
        settings: TensorNetworkSimulationSettings,
    ):  # pragma: no cover
        """Initialize the objective function.

        The :class:`.OneMinusFidelity` class is DEPRECATED since v0.2.
        Please migrate to :class:`.MaximizeStateFidelity` as soon as possible.

        Args:
            ansatz: Parametrized ansatz circuit.
            target: Target state in tensor-network representation.
            settings: Tensor network simulation settings.
        """
        super().__init__(target, ansatz, settings)


# Reminder: update the RST file in docs/apidocs when adding new interfaces.
__all__ = [
    "MaximizeStateFidelity",
    "OneMinusFidelity",
]
