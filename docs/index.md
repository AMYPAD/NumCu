# NumCu

Numerical CUDA-based Python library built on top of [CuVec](https://amypad.github.com/CuVec).

[![Version](https://img.shields.io/pypi/v/o.svg?logo=python&logoColor=white)](https://github.com/AMYPAD/NumCu/releases)
[![Downloads](https://img.shields.io/pypi/dm/numcu.svg?logo=pypi&logoColor=white&label=PyPI%20downloads)](https://pypi.org/project/numcu)
[![Py-Versions](https://img.shields.io/pypi/pyversions/numcu.svg?logo=python&logoColor=white)](https://pypi.org/project/numcu)
[![Licence](https://img.shields.io/pypi/l/numcu.svg?label=licence)](https://github.com/AMYPAD/NumCu/blob/main/LICENCE)
[![Tests](https://img.shields.io/github/workflow/status/AMYPAD/NumCu/Test?logo=GitHub)](https://github.com/AMYPAD/NumCu/actions)
[![Coverage](https://codecov.io/gh/AMYPAD/NumCu/branch/main/graph/badge.svg)](https://codecov.io/gh/AMYPAD/NumCu)

## Install

```sh
pip install numcu
```

Requirements:

- Python 3.6 or greater (e.g. via [Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda))
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
