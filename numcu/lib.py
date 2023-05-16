"""Thin wrappers around `numcu` C++/CUDA module"""
import logging

import cuvec as cu
import numpy as np

from . import numcu as ext  # type: ignore # yapf: disable

__all__ = ['add', 'div', 'mul']
log = logging.getLogger(__name__)
FLOAT_MAX = np.float32(np.inf)


def check_cuvec(a, shape, dtype):
    """Asserts that CuVec `a` is of `shape` & `dtype`"""
    if not isinstance(a, cu.CuVec):
        raise TypeError(f"must be a {cu.CuVec}")
    elif np.dtype(a.dtype) != np.dtype(dtype):
        raise TypeError(f"dtype must be {dtype}: got {a.dtype}")
    elif a.shape != shape:
        raise IndexError(f"shape must be {shape}: got {a.shape}")


def check_similar(*arrays, allow_none=True):
    """Asserts that all arrays are `CuVec`s of the same `shape` & `dtype`"""
    arrs = tuple(filter(lambda x: x is not None, arrays))
    if not allow_none and len(arrays) != len(arrs):
        raise TypeError("must not be None")
    shape, dtype = arrs[0].shape, arrs[0].dtype
    for a in arrs:
        check_cuvec(a, shape, dtype)


def div(numerator, divisor, default=FLOAT_MAX, output=None, dev_id=0, sync=True):
    """
    Elementwise `output = numerator / divisor if divisor else default`
    Args:
      numerator(ndarray): input.
      divisor(ndarray): input.
      default(float): value for zero-division errors.
      output(ndarray): pre-existing output memory.
      dev_id(int or bool): GPU index (`False` for CPU).
      sync(bool): whether to `cudaDeviceSynchronize()` after GPU operations.
    """
    if dev_id is False:
        res = np.divide(numerator, divisor, out=output)
        res[np.isnan(res)] = default
        return res
    cu.dev_set(dev_id)
    numerator = cu.asarray(numerator, 'float32')
    divisor = cu.asarray(divisor, 'float32')
    check_similar(numerator, divisor, output)
    res = ext.div(numerator, divisor, default=default, output=output, log=log.getEffectiveLevel())
    if sync: cu.dev_sync()
    return cu.asarray(res)


def mul(a, b, output=None, dev_id=0, sync=True):
    """
    Elementwise `output = a * b`
    Args:
      a(ndarray): input.
      b(ndarray): input.
      output(ndarray): pre-existing output memory.
      dev_id(int or bool): GPU index (`False` for CPU).
      sync(bool): whether to `cudaDeviceSynchronize()` after GPU operations.
    """
    if dev_id is False: return np.multiply(a, b, out=output)
    cu.dev_set(dev_id)
    a = cu.asarray(a, 'float32')
    b = cu.asarray(b, 'float32')
    check_similar(a, b, output)
    res = ext.mul(a, b, output=output, log=log.getEffectiveLevel())
    if sync: cu.dev_sync()
    return cu.asarray(res)


def add(a, b, output=None, dev_id=0, sync=True):
    """
    Elementwise `output = a + b`
    Args:
      a(ndarray): input.
      b(ndarray): input.
      output(ndarray): pre-existing output memory.
      dev_id(int or bool): GPU index (`False` for CPU).
      sync(bool): whether to `cudaDeviceSynchronize()` after GPU operations.
    """
    if dev_id is False: return np.add(a, b, out=output)
    cu.dev_set(dev_id)
    a = cu.asarray(a, 'float32')
    b = cu.asarray(b, 'float32')
    check_similar(a, b, output)
    res = ext.add(a, b, output=output, log=log.getEffectiveLevel())
    if sync: cu.dev_sync()
    return cu.asarray(res)
