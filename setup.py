from setuptools import setup, find_packages

setup(
    name="DeepGraph",
    version='0.0.6',
    packages=find_packages(),
    author="Dominik Traxl",
    author_email="dominik.traxl@posteo.org",
    url='https://github.com/deepgraph/deepgraph/',
    download_url='https://github.com/deepgraph/deepgraph/tarball/v0.0.6',
    description=("DeepGraph is a scalable, general-purpose data analysis "
                 "package. It implements a network representation based on "
                 "pandas DataFrames and provides methods to construct, "
                 "partition and plot graphs, to interface with popular "
                 "network packages and more."),
    install_requires=['numpy>=1.6',
                      'pandas>=0.14.0'],
    license="BSD",
    classifiers = [
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'],
    package_data={'deepgraph': ['../tests/*.py', '../LICENSE.txt']},
)
