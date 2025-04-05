import sys
import re
from pathlib import Path
from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize


def return_version_string() -> str:
    init_file = Path("src/deepgraph/__init__.py").read_text(encoding="utf-8")
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", init_file, re.MULTILINE)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


extensions = [
    Extension(
        name="deepgraph._triu_indices",
        sources=["src/deepgraph/_triu_indices.pyx"],
        include_dirs=[np.get_include()],
        # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="deepgraph._find_selected_indices",
        sources=["src/deepgraph/_find_selected_indices.pyx"],
        include_dirs=[np.get_include()],
        # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
]

extensions = cythonize(
    extensions,
    compiler_directives={"language_level": sys.version_info[0]}
)

setup(
    version=return_version_string(),
    ext_modules=extensions,
)
