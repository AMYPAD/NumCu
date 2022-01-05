import cuvec as cu
import numpy as np

import numcu as nc


def test_div():
    num, div = cu.asarray(np.random.random((2, 42, 1337, 16)).astype('float32'))
    div -= 0.5
    div[0] = 0
    from warnings import catch_warnings, filterwarnings
    with catch_warnings():
        filterwarnings('ignore', 'divide by zero', RuntimeWarning)
        ref = nc.div(num, div, dev_id=False)
    res = nc.div(num, div)
    assert (res == ref).all()


def test_mul():
    a, b = cu.asarray(np.random.random((2, 42, 1337, 16)).astype('float32') - 0.5)
    ref = nc.mul(a, b, dev_id=False)
    res = nc.mul(a, b)
    assert (res == ref).all()


def test_add():
    a, b = cu.asarray(np.random.random((2, 42, 1337, 16)).astype('float32'))
    ref = nc.add(a, b, dev_id=False)
    res = nc.add(a, b)
    assert (res == ref).all()
