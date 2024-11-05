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

from qiskit_addon_aqc_tensor.simulation.aer import (
    QiskitAerSimulationSettings,
    is_aer_available,
)
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator, is_quimb_available

###
# Perform slow imports lazily as fixtures
###


@pytest.fixture
def AerSimulator():
    from qiskit_aer import AerSimulator

    return AerSimulator


@pytest.fixture
def quimb():
    import quimb.tensor

    return quimb


###
# A fixture that provides all available backends
###


def _aersimulator_factory():
    from qiskit_aer import AerSimulator

    return AerSimulator(method="matrix_product_state")


def _quimb_factory_factory(name, autodiff_backend):
    def _quimb_factory():
        import quimb.tensor as qtn

        return QuimbSimulator(getattr(qtn, name), autodiff_backend)

    _quimb_factory.__name__ = f"{name}_{autodiff_backend}"
    return _quimb_factory


# NOTE: This is all because it is not clear that
# https://docs.pytest.org/en/stable/how-to/fixtures.html#using-marks-with-parametrized-fixtures
# will easily work the way we need it to.  Maybe there is a way, but figuring
# it out is not urgent.
_simulator_factories = []
if is_quimb_available():
    _simulator_factories.append(_quimb_factory_factory("CircuitMPS", "explicit"))
    _simulator_factories.append(_quimb_factory_factory("Circuit", "jax"))
    _simulator_factories.append(_quimb_factory_factory("CircuitPermMPS", "explicit"))
    _simulator_factories.append(_quimb_factory_factory("CircuitPermMPS", "jax"))
if is_aer_available():
    _simulator_factories.append(_aersimulator_factory)
    _simulator_factories.append(lambda: QiskitAerSimulationSettings(_aersimulator_factory()))


@pytest.fixture(params=_simulator_factories)
def available_backend_fixture(request):
    """A fixture that provides all available backends."""
    return request.param()
