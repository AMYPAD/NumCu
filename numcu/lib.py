"""Thin wrappers around `numcu` C++/CUDA module"""
import logging

import cuvec as cu
import numpy as np

from . import numcu as ext  # type: ignore # yapf: disable

__all__ = ['add', 'div', 'mul']
log = logging.getLogger(__name__)
FLOAT_MAX = np.float32(np.inf)


def get_namespace(*xs, default=cu):
    """
    Similar to `array_api_compat.get_namespace`,
    but handles `CuVec`s pretending to be NumPy arrays.
    """
    for a in xs:
        if hasattr(a, 'cuvec'):
            from importlib import import_module

            return import_module(a.__module__)
    return default # backwards compatibility


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
    assert numerator.size == divisor.size
    cu.dev_set(dev_id)
    if output is None:
        output = get_namespace(numerator, divisor, output).zeros_like(numerator)
    ext.div(numerator, divisor, output, default=default)
    if sync: cu.dev_sync()
    return output


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
    assert a.size == b.size
    cu.dev_set(dev_id)
    if output is None:
        output = get_namespace(a, b, output).zeros_like(a)
    ext.mul(a, b, output)
    if sync: cu.dev_sync()
    return output


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
    assert a.size == b.size
    cu.dev_set(dev_id)
    if output is None:
        output = get_namespace(a, b, output).zeros_like(a)
    ext.add(a, b, output)
    if sync: cu.dev_sync()
    return output
