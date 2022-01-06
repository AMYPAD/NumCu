/**
 * Extension module using CuVec.
 *
 * Copyright (2022) Casper da Costa-Luis
 */
#include "Python.h"
#include "elemwise.h"
#include "numcu.h"
#include "pycuvec.cuh" // PyCuVec

static PyObject *elem_div(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyCuVec<float> *src_num = NULL; // numerator
  PyCuVec<float> *src_div = NULL; // divisor
  PyCuVec<float> *dst = NULL;     // output
  float zeroDivDefault = FLOAT_MAX;
  int LOG = LOGDEBUG;

  // Parse the input tuple
  static const char *kwds[] = {"num", "div", "default", "output", "log", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|fOi", (char **)kwds, &asPyCuVec_f, &src_num,
                                   &asPyCuVec_f, &src_div, &zeroDivDefault, &dst, &LOG))
    return NULL;

  if (src_num->shape.size() != src_div->shape.size()) {
    PyErr_SetString(PyExc_IndexError, "inputs must have same ndim");
    return NULL;
  }
  for (size_t i = 0; i < src_num->shape.size(); i++) {
    if (src_num->shape[i] != src_div->shape[i]) {
      PyErr_SetString(PyExc_IndexError, "inputs must have same shape");
      return NULL;
    }
  }

  dst = asPyCuVec(dst);
  if (dst) {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> using provided output\n");
    Py_INCREF((PyObject *)dst); // anticipating returning
  } else {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> creating output image\n");
    dst = PyCuVec_zeros_like(src_num);
    if (!dst) return NULL;
  }

  d_div(dst->vec.data(), src_num->vec.data(), src_div->vec.data(), dst->vec.size(),
        zeroDivDefault);
  return CUDA_PyErr() ? NULL : (PyObject *)dst;
}

static PyObject *elem_mul(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyCuVec<float> *src_a = NULL; // input A
  PyCuVec<float> *src_b = NULL; // input B
  PyCuVec<float> *dst = NULL;   // output
  int LOG = LOGDEBUG;

  // Parse the input tuple
  static const char *kwds[] = {"a", "b", "output", "log", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|Oi", (char **)kwds, &asPyCuVec_f, &src_a,
                                   &asPyCuVec_f, &src_b, &dst, &LOG))
    return NULL;

  if (src_a->shape.size() != src_b->shape.size()) {
    PyErr_SetString(PyExc_IndexError, "inputs must have same ndim");
    return NULL;
  }
  for (size_t i = 0; i < src_a->shape.size(); i++) {
    if (src_a->shape[i] != src_b->shape[i]) {
      PyErr_SetString(PyExc_IndexError, "inputs must have same shape");
      return NULL;
    }
  }

  dst = asPyCuVec(dst);
  if (dst) {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> using provided output\n");
    Py_INCREF((PyObject *)dst); // anticipating returning
  } else {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> creating output image\n");
    dst = PyCuVec_zeros_like(src_a);
    if (!dst) return NULL;
  }

  d_mul(dst->vec.data(), src_a->vec.data(), src_b->vec.data(), dst->vec.size());
  return CUDA_PyErr() ? NULL : (PyObject *)dst;
}

static PyObject *elem_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyCuVec<float> *src_a = NULL; // input A
  PyCuVec<float> *src_b = NULL; // input B
  PyCuVec<float> *dst = NULL;   // output
  int LOG = LOGDEBUG;

  // Parse the input tuple
  static const char *kwds[] = {"a", "b", "output", "log", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|Oi", (char **)kwds, &asPyCuVec_f, &src_a,
                                   &asPyCuVec_f, &src_b, &dst, &LOG))
    return NULL;

  if (src_a->shape.size() != src_b->shape.size()) {
    PyErr_SetString(PyExc_IndexError, "inputs must have same ndim");
    return NULL;
  }
  for (size_t i = 0; i < src_a->shape.size(); i++) {
    if (src_a->shape[i] != src_b->shape[i]) {
      PyErr_SetString(PyExc_IndexError, "inputs must have same shape");
      return NULL;
    }
  }

  dst = asPyCuVec(dst);
  if (dst) {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> using provided output\n");
    Py_INCREF((PyObject *)dst); // anticipating returning
  } else {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> creating output image\n");
    dst = PyCuVec_zeros_like(src_a);
    if (!dst) return NULL;
  }

  d_add(dst->vec.data(), src_a->vec.data(), src_b->vec.data(), dst->vec.size());
  return CUDA_PyErr() ? NULL : (PyObject *)dst;
}

static PyMethodDef numcu_methods[] = {
    {"div", (PyCFunction)elem_div, METH_VARARGS | METH_KEYWORDS, "Elementwise division."},
    {"mul", (PyCFunction)elem_mul, METH_VARARGS | METH_KEYWORDS, "Elementwise multiplication."},
    {"add", (PyCFunction)elem_add, METH_VARARGS | METH_KEYWORDS, "Elementwise addition."},
    {NULL, NULL, 0, NULL} // Sentinel
};

/** module */
static struct PyModuleDef numcu = {PyModuleDef_HEAD_INIT,
                                   "numcu", // module
                                   "NumCu external module.",
                                   -1, // module keeps state in global variables
                                   numcu_methods};
PyMODINIT_FUNC PyInit_numcu(void) {
  Py_Initialize();
  return PyModule_Create(&numcu);
}
