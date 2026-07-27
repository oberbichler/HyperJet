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

    // Indexed scalars carry their own order, so d() and dd() can read the
    // gradient and Hessian directly. Named scalars have no canonical order --
    // each value carries whichever names it picked up -- so the caller has to
    // supply one via names=, which is also what makes a set of values with
    // differing names project onto a single shared layout.
    py::exec(R"PY(
def _indexed_g(v):
    if hasattr(v, "names"):
        raise TypeError(
            "hj.d needs an order for named scalars: pass names=[...]"
        )
    return v.g if hasattr(v, "g") else np.zeros(0)


def _indexed_hm(v):
    if hasattr(v, "names"):
        raise TypeError(
            "hj.dd needs an order for named scalars: pass names=[...]"
        )
    return v.hm() if hasattr(v, "hm") else np.zeros((0, 0))


_f = np.vectorize(lambda v: v.f if hasattr(v, "f") else v)
_g = np.vectorize(_indexed_g, signature="()->(n)")
_hm = np.vectorize(_indexed_hm, signature="()->(n,m)")
_named_g = np.vectorize(lambda v, n: v.g(n), signature="()->(n)", excluded={1})
_named_hm = np.vectorize(lambda v, n: v.hm(n), signature="()->(n,m)", excluded={1})


def f(values):
    return _f(values)


def d(values, names=None):
    if names is None:
        return _g(values)
    return _named_g(values, list(names))


def dd(values, names=None):
    if names is None:
        return _hm(values)
    return _named_hm(values, list(names))
)PY",
             global);

    m.attr("f") = global["f"];
    m.attr("d") = global["d"];
    m.attr("dd") = global["dd"];

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
