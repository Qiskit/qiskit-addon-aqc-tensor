[![Release](https://img.shields.io/pypi/v/qiskit-addon-aqc-tensor.svg?label=Release)](https://github.com/Qiskit/qiskit-addon-aqc-tensor/releases)
![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
[![Python](https://img.shields.io/pypi/pyversions/qiskit-addon-aqc-tensor?label=Python&logo=python)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%201.2%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
[![Docs (stable)](https://img.shields.io/badge/%F0%9F%93%84%20Docs-stable-blue.svg)](https://qiskit.github.io/qiskit-addon-aqc-tensor/)
[![License](https://img.shields.io/github/license/Qiskit/qiskit-addon-aqc-tensor?label=License)](LICENSE.txt)
[![Downloads](https://img.shields.io/pypi/dm/qiskit-addon-aqc-tensor.svg?label=Downloads)](https://pypi.org/project/qiskit-addon-aqc-tensor/)
[![Tests](https://github.com/Qiskit/qiskit-addon-aqc-tensor/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit/qiskit-addon-aqc-tensor/actions/workflows/test_latest_versions.yml)
[![Coverage](https://coveralls.io/repos/github/Qiskit/qiskit-addon-aqc-tensor/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-addon-aqc-tensor?branch=main)

# Qiskit addon: approximate quantum compilation with tensor networks (AQC-Tensor)

### Table of contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Deprecation Policy](#deprecation-policy)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)

----------------------------------------------------------------------------------------------------

### About

[Qiskit addons](https://docs.quantum.ibm.com/guides/addons) are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This addon enables a Qiskit user to perform approximate quantum compilation using tensor networks,
a technique that was introduced in [arXiv:2301.08609](https://arxiv.org/abs/2301.08609).

Specifically, this package allows one to compile the _initial portion_ of a circuit into a nearly equivalent approximation of that circuit, but with much fewer layers.

It has been tested primarily on Trotter circuits to date.  It may, however, be applicable to any class of circuits where one has access to both:

1. A _great_ intermediate state, known as the "target state," that can be achieved by tensor-network simulation; and,
2. A _good_ circuit that prepares an approximation to the target state, but with fewer layers when compiled to the target hardware device.

![Compression of initial portion of circuit with AQC](docs/images/aqc-compression.png)

(Figure is taken from [arXiv:2301.08609](https://arxiv.org/abs/2301.08609).)

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at https://qiskit.github.io/qiskit-addon-aqc-tensor/.

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible.

To be useful, this package requires at least one tensor-network backend.  The following command installs the [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) backend, as well as the [quimb](https://github.com/jcmgray/quimb) backend with automatic differentiation support from [JAX](https://github.com/jax-ml/jax):

```bash
pip install 'qiskit-addon-aqc-tensor[aer,quimb-jax]'
```

For more installation information refer to these [installation instructions](INSTALL.rst).

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
[release notes](https://qiskit.github.io/qiskit-addon-aqc-tensor/release-notes.html).

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-addon-aqc-tensor).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-addon-aqc-tensor/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-addon-aqc-tensor/issues/new/choose) for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### Citation

If you use this package in your research, please cite it according to the [CITATION.bib](https://github.com/Qiskit/qiskit-addon-aqc-tensor/blob/main/CITATION.bib) file.

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)
