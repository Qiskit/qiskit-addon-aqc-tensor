Installation instructions
=========================

There are two primary ways to do run and install the packages:

- :ref:`Option 1`
- :ref:`Option 2`

Pre-installation
^^^^^^^^^^^^^^^^

To install locally (using either :ref:`Option 1` or :ref:`Option 2`),
follow a brief set of common instructions to prepare a Python environment for
installation of AQC-Tensor:

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate

Note: If you are using Windows, use the following commands in PowerShell:

.. code:: pwsh

    python3 -m venv c:\path\to\virtual\environment
    c:\path\to\virtual\environment\Scripts\Activate.ps1


.. _Option 1:

Option 1: Pip installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Upgrade pip and install the AQC-Tensor package.  To meaningfully use the package, you must also install at least one tensor network backend.  The following snippet installs the addon, along with quimb (for tensor network support) and jax (for automatic differentiation).

.. code:: sh

    pip install --upgrade pip
    pip install 'qiskit-addon-aqc-tensor[quimb-jax]'


.. _Option 2:

Option 2: Install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To develop in the repository or to run the tutorials locally, install from source:

In either case, the first step is to clone the AQC-Tensor repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-addon-aqc-tensor.git

Next, upgrade pip and enter the repository.

.. code:: sh

    pip install --upgrade pip
    cd qiskit-addon-aqc-tensor

The next step is to install AQC-Tensor to the virtual environment. If you plan on running the tutorials, install the
notebook dependencies in order to run all the visualizations in the notebooks.
If you plan on developing in the repository, install the ``dev`` dependencies.

Adjust the options below to suit your needs.

.. code:: sh

    pip install tox jupyterlab -e '.[notebook-dependencies,dev]'

If you installed the notebook dependencies, you can get started with AQC-Tensor by running the notebooks in the docs.

.. code::

    cd docs/
    jupyter lab


.. _Platform Support:

Platform support
^^^^^^^^^^^^^^^^

We expect this package to work on `any Tier 1 platform supported by Qiskit <https://quantum.cloud.ibm.com/docs/guides/install-qiskit#operating-system-support>`__.
