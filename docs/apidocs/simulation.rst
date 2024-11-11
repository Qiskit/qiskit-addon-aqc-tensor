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

Functions
=========

The following functions provide a common interface that can be used with any supported backend. Regardless of which backend you use, you will invoke these functions as a common interface.

In each function below, the documentation shows every distinct implementation available for that function. The addon will choose which implementation to use based on the types of the arguments that you provide. (This paradigm is called "multiple dispatch".)

.. autofunction:: tensornetwork_from_circuit
.. autofunction:: apply_circuit_to_state
.. autofunction:: compute_overlap

Abstract classes
================

These common classes are used by different backends to TODO...

.. autoclass:: TensorNetworkState
.. autoclass:: TensorNetworkSimulationSettings
