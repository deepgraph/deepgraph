import sys
from setuptools import setup, find_packages, Extension
import numpy as np

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'
# cppext = '' if USE_CYTHON else 'pp'

extensions = [
    Extension(
        "deepgraph._triu_indices",
        ["deepgraph/_triu_indices" + ext],
        include_dirs=[np.get_include()],
        # language='c++',
    ),
    Extension(
        "deepgraph._find_selected_indices",
        ["deepgraph/_find_selected_indices" + ext],
        include_dirs=[np.get_include()]
    )
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(
        extensions,
        compiler_directives={'language_level': sys.version_info[0]}
    )

setup(
    name="DeepGraph",
    version='0.2.3',
    packages=find_packages(),
    author="Dominik Traxl",
    author_email="dominik.traxl@posteo.org",
    url='https://github.com/deepgraph/deepgraph/',
    download_url='https://github.com/deepgraph/deepgraph/tarball/v0.2.3',
    description=("Analyze Data with Pandas-based Networks."),
    long_description=open('README.rst').read(),
    install_requires=['numpy>=1.6',
                      'pandas>=0.17.0'],
    license="BSD",
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Cython',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'],
    ext_modules=extensions,
    package_data={'deepgraph': ['../tests/*.py',
                                '../LICENSE.txt',
                                './*.pyx',
                                './*.c',
                                './*.cpp',
                                ]},
)
