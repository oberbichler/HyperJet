#pragma once

#include <hyperjet.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

namespace hj = pybind11;

namespace py = pybind11;
using namespace py::literals;

template <typename T>
auto bind(py::module &m, const std::string &name)
{
    py::class_<T> cls(m, name.c_str());

    return cls;
}