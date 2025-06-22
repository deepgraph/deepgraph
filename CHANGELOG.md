# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
since version 1.1.0.


## [Unreleased] - 2025-06-22

### Added

### Changed

### Removed

### Fixed


## [1.1.0] - 2025-06-22

This release adds support for new, valid return 
formats of user-defined connector/selector functions (see 
[create_edges](https://deepgraph.readthedocs.io/en/latest/api_reference.html#creating-edges)).
Specifically, explicit tuples and multi-line return statements are now possible. 
This enhancement increases flexibility and improves input validation. 

The public API remains unchanged.

### Added

- Introduced a robust AST-based implementation for extracting output variables from user-defined functions.
  - This improves reliability and parsing accuracy across supported Python versions (3.9+).
  - The extraction now strictly enforces that return statements contain only variable names.
  - It is now also possible to return output variables as:

    **Example: explicit tuple**

    ```python
    def velocity(dt, x_s, x_t):
        dx = x_t - x_s
        v = dx / dt
        return (v, dx)
    ```

    **Example: multi-line tuple**

    ```python
    def velocity(dt, x_s, x_t):
        dx = x_t - x_s
        v = dx / dt
        return (
            v,
            dx
        )
    ```
- Added CHANGELOG.md file.


### Changed

- Replaced previous string-based return parsing using `inspect.getsourcelines()` with a more 
  accurate approach using `inspect.getsource()` and the `ast` module.
- Output validation errors will now be raised if the return statement contains expressions, 
  function calls, or anything other than variable names.
- Updated input argument parsing logic to use `inspect.signature()` instead of 
  `inspect.getfullargspec()`, improving forward compatibility.


### Removed

- The conda recipe to build a conda package locally. Conda packages are build by conda-forge 
  using [this recipe](https://github.com/conda-forge/deepgraph-feedstock/blob/main/recipe/meta.yaml).


### Fixed

- Resolved "UnclosedFileWarning: Closing remaining open file" warnings in tests involving hdf tables.
- Homogenized license information across project files
- `pyproject.toml` had wrong license information (BSD-2-Clause), changed it to the correct one (BSD-3-Clause).