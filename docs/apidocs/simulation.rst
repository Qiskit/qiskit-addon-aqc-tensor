#####################################################################
Tensor Network Simulation (:mod:`qiskit_addon_aqc_tensor.simulation`)
#####################################################################

.. automodule:: qiskit_addon_aqc_tensor.simulation
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: qiskit_addon_aqc_tensor.simulation

Available backends
==================

* :mod:`qiskit_addon_aqc_tensor.simulation.aer`
* :mod:`qiskit_addon_aqc_tensor.simulation.quimb`

Abstract classes
================

.. autoclass:: TensorNetworkState
.. autoclass:: TensorNetworkSimulationSettings

Abstract functions
==================

The following abstract functions provide a common interface that can be used with any supported backend:

.. autofunction:: tensornetwork_from_circuit
.. autofunction:: apply_circuit_to_state
.. autofunction:: compute_overlap
