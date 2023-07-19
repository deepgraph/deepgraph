import sys
from setuptools import setup, find_packages, Extension
import numpy as np
from Cython.Build import cythonize

extensions = [
    Extension(
        name="deepgraph._triu_indices",
        sources=["deepgraph/_triu_indices.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="deepgraph._find_selected_indices",
        sources=["deepgraph/_find_selected_indices.pyx"],
        include_dirs=[np.get_include()],
    ),
]

extensions = cythonize(extensions, compiler_directives={"language_level": sys.version_info[0]})

setup(
    name="DeepGraph",
    version="0.2.3",
    packages=find_packages(),
    author="Dominik Traxl",
    author_email="dominik.traxl@posteo.org",
    url="https://github.com/deepgraph/deepgraph/",
    download_url="https://github.com/deepgraph/deepgraph/tarball/v0.2.3",
    description=("Analyze Data with Pandas-based Networks."),
    long_description=open("README.rst").read(),
    install_requires=["numpy>=1.6", "pandas>=0.17.0"],
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Cython",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    ext_modules=extensions,
    package_data={
        "deepgraph": [
            "../tests/*.py",
            "../LICENSE.txt",
            "./*.pyx",
            "./*.c",
        ]
    },
)
