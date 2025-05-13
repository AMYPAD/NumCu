/**
 * Extension module using CuVec.
 *
 * Copyright (2022) Casper da Costa-Luis
 */
#include "elemwise.h"          // div, mul, add
#include <nanobind/nanobind.h> // nanobind, NB_MODULE
#include <nanobind/ndarray.h>  // ndarray
#include <pycuvec.cuh>         // CUDA_PyErr

namespace nb = nanobind;
template <typename T> using Arr = const nb::ndarray<T>;

template <typename T>
void elem_div(Arr<const T> &num, Arr<const T> &den, Arr<T> &dst, T zeroDivDefault) {
  div(dst.data(), num.data(), den.data(), dst.size(), zeroDivDefault);
  if (CUDA_PyErr()) throw std::runtime_error("CUDA kernel");
}

template <typename T> void elem_mul(Arr<const T> &a, Arr<const T> &b, Arr<T> &dst) {
  mul(dst.data(), a.data(), b.data(), dst.size());
  if (CUDA_PyErr()) throw std::runtime_error("CUDA kernel");
}

template <typename T> void elem_add(Arr<const T> &a, Arr<const T> &b, Arr<T> &dst) {
  add(dst.data(), a.data(), b.data(), dst.size());
  if (CUDA_PyErr()) throw std::runtime_error("CUDA kernel");
}

using namespace nb::literals;
NB_MODULE(numcu, m) {
  m.doc() = "NumCu external module.";
  m.def("div", &elem_div<float>, "Elementwise division.", "numerator"_a, "divisor"_a, "output"_a,
        "default"_a = FLOAT_MAX);
  m.def("mul", &elem_mul<float>, "Elementwise multiplication.", "a"_a, "b"_a, "output"_a);
  m.def("add", &elem_add<float>, "Elementwise addition.", "a"_a, "b"_a, "output"_a);
}
