from setuptools import setup, find_packages

setup(
    name="DeepGraph",
    version='0.0.1',
    packages=find_packages(),
    author="Dominik Traxl",
    author_email="dominik.traxl@posteo.org",
    url='https://github.com/deepgraph/deepgraph/',
    download_url='https://github.com/deepgraph/deepgraph/tarball/0.0.1',
    description=("DeepGraph is an efficient, general-purpose data analysis "
                 "Python package. Based on pandas DataFrames, it provides "
                 "the means to analyze data via graph theory "
                 "(a.k.a. networks)."),
    install_requires=['numpy>=1.6',
                      'pandas>=0.14.0'],
)
