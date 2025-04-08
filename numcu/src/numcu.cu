/**
 * Extension module using CuVec.
 *
 * Copyright (2022) Casper da Costa-Luis
 */
#include "elemwise.h"          // div, mul, add
#include <pybind11/pybind11.h> // pybind11
#include <pycuvec.cuh>         // CUDA_PyErr

namespace py = pybind11;

template <typename T>
void elem_div(py::buffer num, py::buffer den, py::buffer dst, T zeroDivDefault) {
  py::buffer_info src_num = num.request(), src_den = den.request(), dst_out = dst.request(true);
  div(static_cast<T *>(dst_out.ptr), static_cast<T *>(src_num.ptr), static_cast<T *>(src_den.ptr),
      dst_out.size, zeroDivDefault);
  CUDA_PyErr();
}

template <typename T> void elem_mul(py::buffer a, py::buffer b, py::buffer dst) {
  py::buffer_info src_a = a.request(), src_b = b.request(), dst_out = dst.request(true);
  mul(static_cast<T *>(dst_out.ptr), static_cast<T *>(src_a.ptr), static_cast<T *>(src_b.ptr),
      dst_out.size);
  CUDA_PyErr();
}

template <typename T> void elem_add(py::buffer a, py::buffer b, py::buffer dst) {
  py::buffer_info src_a = a.request(), src_b = b.request(), dst_out = dst.request(true);
  add(static_cast<T *>(dst_out.ptr), static_cast<T *>(src_a.ptr), static_cast<T *>(src_b.ptr),
      dst_out.size);
  CUDA_PyErr();
}

using namespace pybind11::literals;
PYBIND11_MODULE(numcu, m) {
  m.doc() = "NumCu external module.";
  m.def("div", &elem_div<float>, "Elementwise division.", "numerator"_a, "divisor"_a, "output"_a,
        "default"_a = FLOAT_MAX);
  m.def("mul", &elem_mul<float>, "Elementwise multiplication.", "a"_a, "b"_a, "output"_a);
  m.def("add", &elem_add<float>, "Elementwise addition.", "a"_a, "b"_a, "output"_a);
}
