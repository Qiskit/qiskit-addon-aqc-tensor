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

# pylint: disable=invalid-name

"""
Sphinx documentation builder
"""

# General options:
import inspect
import os
import re
import sys
from pathlib import Path

import qiskit_addon_aqc_tensor

project = "AQC-Tensor"
copyright = "2024"  # pylint: disable=redefined-builtin
author = "IBM Quantum"

_rootdir = Path(__file__).parent.parent
sys.path.insert(0, str(_rootdir))

# The full version, including alpha/beta/rc tags
release = qiskit_addon_aqc_tensor.__version__
# The short X.Y version
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "matplotlib.sphinxext.plot_directive",
    # "sphinx.ext.autosectionlabel",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "reno.sphinxext",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "sphinx.ext.intersphinx",
    "qiskit_sphinx_theme",
]
numfig = False
numfig_format = {"table": "Table %s"}
language = "en"
pygments_style = "colorful"
add_module_names = False
modindex_common_prefix = ["qiskit_addon_aqc_tensor."]

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
html_theme_options = {
    "dark_logo": "images/qiskit-dark-logo.svg",
    "light_logo": "images/qiskit-light-logo.svg",
    "sidebar_qiskit_ecosystem_member": False,
}
html_static_path = ["_static"]
templates_path = ["_templates"]

# Options for autodoc. These reflect the values from Qiskit SDK and Runtime.
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {
    "inherited-members": None,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False


# nbsphinx options (for tutorials)
nbsphinx_timeout = 180
nbsphinx_execute = "always" if os.environ.get("CI") == "true" else "auto"
nbsphinx_widgets_path = ""
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "test_notebooks",
    "**/README.rst",
]

# matplotlib.sphinxext.plot_directive options
plot_html_show_formats = False
plot_formats = ["svg"]

# ----------------------------------------------------------------------------------
# Redirects
# ----------------------------------------------------------------------------------

_inlined_apis = [
    ("qiskit_addon_aqc_tensor.objective", "OneMinusFidelity"),
    ("qiskit_addon_aqc_tensor.simulation", "TensorNetworkState"),
    ("qiskit_addon_aqc_tensor.simulation", "TensorNetworkSimulationSettings"),
    ("qiskit_addon_aqc_tensor.simulation", "tensornetwork_from_circuit"),
    ("qiskit_addon_aqc_tensor.simulation", "apply_circuit_to_state"),
    ("qiskit_addon_aqc_tensor.simulation", "compute_overlap"),
]

redirects = {
    "stubs/qiskit_addon_aqc_tensor.ansatz_generation.generate_ansatz_from_circuit": "../apidocs/qiskit-addon-aqc-tensor.html#qiskit_addon_aqc_tensor.generate_ansatz_from_circuit",
    **{
        f"stubs/{module}.{name}": f"../apidocs/{module.split('.')[-1]}.html#{module}.{name}"
        for module, name in _inlined_apis
    },
}


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit/", None),
    "qiskit-ibm-runtime": (
        "https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/",
        None,
    ),
    "qiskit-aer": ("https://qiskit.github.io/qiskit-aer/", None),
    "rustworkx": ("https://www.rustworkx.org/", None),
    "qiskit_addon_utils": ("https://docs.quantum.ibm.com/api/qiskit-addon-utils/", None),
    "quimb": ("https://quimb.readthedocs.io/en/latest/", None),
}

# ----------------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------------


def determine_github_branch() -> str:
    """Determine the GitHub branch name to use for source code links.

    We need to decide whether to use `stable/<version>` vs. `main` for dev builds.
    Refer to https://docs.github.com/en/actions/learn-github-actions/variables
    for how we determine this with GitHub Actions.
    """
    # If CI env vars not set, default to `main`. This is relevant for local builds.
    if "GITHUB_REF_NAME" not in os.environ:
        return "main"

    # PR workflows set the branch they're merging into.
    if base_ref := os.environ.get("GITHUB_BASE_REF"):
        return base_ref

    ref_name = os.environ["GITHUB_REF_NAME"]

    # Check if the ref_name is a tag like `1.0.0` or `1.0.0rc1`. If so, we need
    # to transform it to a Git branch like `stable/1.0`.
    version_without_patch = re.match(r"(\d+\.\d+)", ref_name)
    return f"stable/{version_without_patch.group()}" if version_without_patch else ref_name


GITHUB_BRANCH = determine_github_branch()


def linkcode_resolve(domain, info):
    """Add links to GitHub source code."""
    if domain != "py":
        return None

    module_name = info["module"]
    module = sys.modules.get(module_name)
    if module is None or "qiskit_addon_aqc_tensor" not in module_name:
        return None

    def is_valid_code_object(obj):
        return inspect.isclass(obj) or inspect.ismethod(obj) or inspect.isfunction(obj)

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
        if not is_valid_code_object(obj):
            return None

    # Unwrap decorators. This requires they used `functools.wrap()`.
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
        if not is_valid_code_object(obj):
            return None

    try:
        full_file_name = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if full_file_name is None or "/qiskit_addon_aqc_tensor/" not in full_file_name:
        return None
    file_name = full_file_name.split("/qiskit_addon_aqc_tensor/")[-1]

    try:
        source, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        linespec = ""
    else:
        ending_lineno = lineno + len(source) - 1
        linespec = f"#L{lineno}-L{ending_lineno}"
    return f"https://github.com/Qiskit/qiskit-addon-aqc-tensor/tree/{GITHUB_BRANCH}/qiskit_addon_aqc_tensor/{file_name}{linespec}"
