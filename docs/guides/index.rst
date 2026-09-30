######
Guides
######

This page summarizes the guides that are available for AQC-Tensor.  Each one focuses on a specific aspect of
the package.  For an end-to-end workflow that executes on quantum hardware, see the
`AQC-Tensor tutorial <https://quantum.cloud.ibm.com/docs/tutorials/approximate-quantum-compilation-for-time-evolution>`__
hosted on the IBM Quantum Platform.

Getting started
---------------

- :doc:`Compress a deep Trotter circuit into a shallower ansatz, in a minimal working example <quickstart>`

Beyond the basics
-----------------

- :doc:`Understand how AQC-Tensor works, including ansatz generation, the objective function, and the available tensor-network backends <explanation>`
- :doc:`Work with quimb's TNOptimizer directly, for more control than the built-in backend provides <01_quimb_tnoptimizer>`

.. toctree::
   :hidden:
   :caption: Getting started

   self
   Quick start <quickstart>

.. toctree::
   :hidden:
   :caption: Beyond the basics

   Explanatory material <explanation>
   Use quimb's TNOptimizer directly <01_quimb_tnoptimizer>
