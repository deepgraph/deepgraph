from setuptools import setup, find_packages

setup(
    name="DeepGraph",
    version='0.0.2',
    packages=find_packages(),
    author="Dominik Traxl",
    author_email="dominik.traxl@posteo.org",
    url='https://github.com/deepgraph/deepgraph/',
    download_url='https://github.com/deepgraph/deepgraph/tarball/0.0.2',
    description=("DeepGraph is an efficient, general-purpose data analysis "
                 "Python package. Based on pandas DataFrames, it provides "
                 "the means to analyze data via graph theory "
                 "(a.k.a. networks)."),
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
