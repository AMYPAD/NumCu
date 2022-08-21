NumCu
=====

Numerical CUDA-based Python library built on top of `CuVec <https://github.com/AMYPAD/CuVec>`_.

|Version| |Downloads| |Py-Versions| |DOI| |Licence| |Tests| |Coverage|

.. contents:: Table of contents
   :backlinks: top
   :local:

Install
~~~~~~~

Requirements:

- Python 3.6 or greater (e.g. via `Anaconda or Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda>`_ or via `python3-dev`)
- (optional) `CUDA SDK/Toolkit <https://developer.nvidia.com/cuda-downloads>`_ (including drivers for an NVIDIA GPU)

  * note that if the CUDA SDK/Toolkit is installed *after* NumCu, then NumCu must be re-installed to enable CUDA support

.. code:: sh

    pip install numcu

Usage
~~~~~

See `the usage documentation <https://amypad.github.io/NumCu/#usage>`_.

Contributing
~~~~~~~~~~~~

See `CONTRIBUTING.md <https://github.com/AMYPAD/NumCu/blob/main/CONTRIBUTING.md>`_.

Licence
~~~~~~~

|Licence| |DOI|

Copyright 2022

- `Casper O. da Costa-Luis <https://github.com/casperdcl>`__ @ University College London/King's College London
- `Contributors <https://github.com/AMYPAD/numcu/graphs/contributors>`__

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7013340.svg
   :target: https://doi.org/10.5281/zenodo.7013340
.. |Licence| image:: https://img.shields.io/pypi/l/numcu.svg?label=licence
   :target: https://github.com/AMYPAD/NumCu/blob/main/LICENCE
.. |Tests| image:: https://img.shields.io/github/workflow/status/AMYPAD/NumCu/Test?logo=GitHub
   :target: https://github.com/AMYPAD/NumCu/actions
.. |Downloads| image:: https://img.shields.io/pypi/dm/numcu.svg?logo=pypi&logoColor=white&label=PyPI%20downloads
   :target: https://pypi.org/project/numcu
.. |Coverage| image:: https://codecov.io/gh/AMYPAD/NumCu/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/AMYPAD/NumCu
.. |Version| image:: https://img.shields.io/pypi/v/numcu.svg?logo=python&logoColor=white
   :target: https://github.com/AMYPAD/NumCu/releases
.. |Py-Versions| image:: https://img.shields.io/pypi/pyversions/numcu.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/numcu
