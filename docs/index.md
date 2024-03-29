# NumCu

Numerical CUDA-based Python library built on top of [CuVec](https://amypad.github.com/CuVec).

[![Version](https://img.shields.io/pypi/v/numcu.svg?logo=python&logoColor=white)](https://github.com/AMYPAD/NumCu/releases)
[![Downloads](https://img.shields.io/pypi/dm/numcu.svg?logo=pypi&logoColor=white&label=PyPI%20downloads)](https://pypi.org/project/numcu)
[![Py-Versions](https://img.shields.io/pypi/pyversions/numcu.svg?logo=python&logoColor=white)](https://pypi.org/project/numcu)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7013340.svg)](https://doi.org/10.5281/zenodo.7013340)
[![Licence](https://img.shields.io/pypi/l/numcu.svg?label=licence)](https://github.com/AMYPAD/NumCu/blob/main/LICENCE)
[![Tests](https://img.shields.io/github/actions/workflow/status/AMYPAD/NumCu/test.yml?branch=main&logo=GitHub)](https://github.com/AMYPAD/NumCu/actions)
[![Coverage](https://codecov.io/gh/AMYPAD/NumCu/branch/main/graph/badge.svg)](https://codecov.io/gh/AMYPAD/NumCu)

## Install

```sh
pip install numcu
```

Requirements:

- Python 3.7 or greater (e.g. via [Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda) or via `python3-dev`)
- (optional) [CUDA SDK/Toolkit](https://developer.nvidia.com/cuda-downloads) (including drivers for an NVIDIA GPU)
  + note that if the CUDA SDK/Toolkit is installed *after* NumCu, then NumCu must be re-installed to enable CUDA support

## Usage

```py
import numcu as nc
import numpy as np

a = nc.zeros((1337, 42), "float32")
assert isinstance(cu.zeros(1), np.ndarray)

b = nc.ones_like(a)
assert np.all(nc.div(a, b) == 0)
```
