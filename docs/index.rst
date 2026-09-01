#################################################################
Approximate quantum compilation with tensor networks (AQC-Tensor)
#################################################################


With the AQC-Tensor addon, you can perform approximate quantum compilation by using tensor networks,
a technique that was introduced in `arXiv:2301.08609 <https://arxiv.org/abs/2301.08609>`__.

Specifically, with this package you can compile the *initial portion* of a circuit into a nearly equivalent approximation of that circuit, but with fewer layers.

It has been tested primarily on Trotter circuits to date.  It might, however, be applicable to any class of circuits where you have access to both of the following:

- A *great* intermediate state, known as the "target state", that can be achieved by tensor-network simulation
- A *good* circuit that prepares an approximation to the target state, but with fewer layers when compiled to the target hardware device

.. image:: images/aqc-compression.png

(Figure is taken from `arXiv:2301.08609 <https://arxiv.org/abs/2301.08609>`__.)

Technical discussion
--------------------

Ansatz generation
""""""""""""""""""

The :func:`.generate_ansatz_from_circuit` function takes an input circuit and returns a parametrized ansatz circuit (along with initial parameters that reproduce the input circuit, up to a global phase).  The ansatz parametrizes each two-qubit block using the `KAK decomposition <https://qiskit.github.io/qiskit-addon-cutting/explanation/#more-general-cut-two-qubit-gates-via-the-kak-decomposition>`__, which expresses any two-qubit gate in terms of three parameters, up to single-qubit rotations.  See the :ref:`explanatory material <explanation>` for details.

Tensor-network simulation
"""""""""""""""""""""""""

The tensor-network simulation is delegated to a pluggable backend; this package currently supports Qiskit Aer's MPS simulator and quimb's simulators.  Each backend also provides a means of evaluating the gradient of the objective function --- either an explicit gradient or, in the case of quimb, an automatic-differentiation backend.  See the :ref:`explanatory material <explanation>` for the available choices.

The most important parameter of a tensor network is its maximum bond dimension, which limits how much entanglement it can represent (and thus to what depth a given circuit can be faithfully simulated).  The bond dimension is often represented by the Greek letter :math:`\chi`.

Given a general circuit on :math:`L` qubits, a matrix-product state needs at most a bond dimension of :math:`\chi_\mathrm{exact} = 2^{\lfloor L/2 \rfloor}` to be able to simulate it to *any* depth.  Of course, general circuits on 100+ qubits cannot be classically simulated, so it will be intractable to set the bond dimension this high for those circuits.

For this reason, if you are attempting to experiment with AQC-Tensor on a toy problem with few qubits, it is important to ensure that :math:`\chi < 2^{\lfloor L/2 \rfloor}`.  Otherwise, any circuit can be simulated to any depth, and there is no point in performing AQC.

Optimization method
"""""""""""""""""""

Users are encouraged to use :mod:`scipy.optimize` to perform the optimization.

L-BFGS is the optimizer demonstrated in the tutorial notebook. It works well in practice because it uses the function value and its gradient to approximate the Hessian.  It works well when given an initial point and seems to work particularly well in the case of Trotter circuits.  However, it might terminate early if it starts in a barren plateau.  In that case, performing a handful of steps using the ADAM optimizer first might help.

Developer guide
---------------

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-aqc-tensor/blob/main/CONTRIBUTING.md>`__ in the root of this project's repository.

Citing this project
-------------------

If you use this package in your research, use the ``CITATON.bib`` file in this project's repository to cite the appropriate references:

.. literalinclude:: ../CITATION.bib
   :language: bibtex


.. toctree::
  :hidden:

   Home <self>
   Installation instructions <install>
   Guides <guides/index>
   GitHub <https://github.com/Qiskit/qiskit-addon-aqc-tensor>

.. toctree::
  :hidden:
  :caption: Tutorials

   Approximate quantum compilation for time evolution circuits <https://quantum.cloud.ibm.com/docs/en/tutorials/approximate-quantum-compilation-for-time-evolution>

.. toctree::
  :hidden:
  :caption: API reference

   Python API reference <apidocs/index>
   Release notes <release-notes>
