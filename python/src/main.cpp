// this unit imports the numpy C API tables that dtype.h shares
#define HYPERJET_IMPORT_ARRAY

#include "common.h"

// generated, see python/CMakeLists.txt
void bind_ddscalar_all(pybind11::module &);
void variables_dispatch(pybind11::list &, const std::vector<double> &,
                        const hj::index);

void bind_sscalar(pybind11::module &);

PYBIND11_MODULE(hyperjet, m) {
  using namespace pybind11::literals;

  namespace py = pybind11;
  namespace hj = hyperjet;

  if (_import_array() < 0 || _import_umath() < 0) {
    throw py::error_already_set();
  }

  m.doc() = "HyperJet by Thomas Oberbichler";
  m.attr("__author__") = "Thomas Oberbichler";
  m.attr("__copyright__") = "Copyright (c) 2019-2021, Thomas Oberbichler";
  m.attr("__version__") = HYPERJET_VERSION;
  m.attr("__email__") = "thomas.oberbichler@gmail.com";
  m.attr("__status__") = "Development";

  bind_ddscalar_all(m);

  bind_sscalar(m);

  // utilities
  {
    py::object numpy = py::module::import("numpy");
    auto global = py::dict();
    global["np"] = numpy;

    m.attr("f") = py::eval(
        "np.vectorize(lambda v: v.f if hasattr(v, 'f') else v)", global);
    m.attr("d") = py::eval("np.vectorize(lambda v: v.g if hasattr(v, 'g') else "
                           "np.zeros((0)), signature='()->(n)')",
                           global);
    m.attr("dd") = py::eval("np.vectorize(lambda v: v.hm() if hasattr(v, 'hm') "
                            "else np.zeros((0, 0)), signature='()->(n,m)')",
                            global);

    m.def(
        "variables",
        [](const std::vector<double> &values, const hj::index order) {
          if (order < 0 || 2 < order) {
            throw std::runtime_error("Invalid order");
          }

          py::list results;

          if (order == 0) {
            results.attr("extend")(values);
          } else {
            variables_dispatch(results, values, order);
          }

          return results;
        },
        "values"_a, "order"_a = 2);
  }
}
