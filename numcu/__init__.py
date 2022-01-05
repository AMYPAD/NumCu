"""
Numerical CUDA-based Python library.
"""
__author__ = "Casper O. da Costa-Luis"
__date__ = "2022"
# version detector. Precedence: installed dist, git, 'UNKNOWN'
try:
    from ._dist_ver import __version__
except ImportError: # pragma: nocover
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="../..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "UNKNOWN"
__all__ = [
    # config
    'cmake_prefix', 'include_path',
    # cuvec
    'dev_sync', 'copy', 'asarray',
    'zeros', 'ones', 'zeros_like', 'ones_like',
    # numerical
    'add', 'div', 'mul'] # yapf: disable

try:          # py<3.9
    import importlib_resources as resources
except ImportError:
    from importlib import resources

try:
    from cuvec import dev_sync
except ImportError as err: # pragma: no cover
    from warnings import warn
    warn(str(err), UserWarning)
else:
    from cuvec import asarray, copy, ones, ones_like, zeros, zeros_like

    from .lib import add, div, mul

p = resources.files('numcu').resolve()
# for C++/CUDA/SWIG includes
include_path = p / 'include'
# for use in `cmake -DCMAKE_PREFIX_PATH=...`
cmake_prefix = p / 'cmake'
