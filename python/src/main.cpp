#include "common.h"

void bind_ddscalar_xd(pybind11::module &);
void bind_ddscalar_0d(pybind11::module &);
void bind_ddscalar_1d(pybind11::module &);
void bind_ddscalar_2d(pybind11::module &);
void bind_ddscalar_3d(pybind11::module &);
void bind_ddscalar_4d(pybind11::module &);
void bind_ddscalar_5d(pybind11::module &);
void bind_ddscalar_6d(pybind11::module &);
void bind_ddscalar_7d(pybind11::module &);
void bind_ddscalar_8d(pybind11::module &);
void bind_ddscalar_9d(pybind11::module &);
void bind_ddscalar_10d(pybind11::module &);
void bind_ddscalar_11d(pybind11::module &);
void bind_ddscalar_12d(pybind11::module &);
void bind_ddscalar_13d(pybind11::module &);
void bind_ddscalar_14d(pybind11::module &);
void bind_ddscalar_15d(pybind11::module &);
void bind_ddscalar_16d(pybind11::module &);

void bind_sscalar(pybind11::module &);

PYBIND11_MODULE(hyperjet, m)
{
    using namespace pybind11::literals;

    namespace py = pybind11;
    namespace hj = hyperjet;

    m.doc() = "HyperJet by Thomas Oberbichler";
    m.attr("__author__") = "Thomas Oberbichler";
    m.attr("__copyright__") = "Copyright (c) 2019-2021, Thomas Oberbichler";
    m.attr("__version__") = HYPERJET_VERSION;
    m.attr("__email__") = "thomas.oberbichler@gmail.com";
    m.attr("__status__") = "Development";

    bind_ddscalar_xd(m);
    bind_ddscalar_0d(m);
    bind_ddscalar_1d(m);
    bind_ddscalar_2d(m);
    bind_ddscalar_3d(m);
    bind_ddscalar_4d(m);
    bind_ddscalar_5d(m);
    bind_ddscalar_6d(m);
    bind_ddscalar_7d(m);
    bind_ddscalar_8d(m);
    bind_ddscalar_9d(m);
    bind_ddscalar_10d(m);
    bind_ddscalar_11d(m);
    bind_ddscalar_12d(m);
    bind_ddscalar_13d(m);
    bind_ddscalar_14d(m);
    bind_ddscalar_15d(m);
    bind_ddscalar_16d(m);

    bind_sscalar(m);

    // utilities
    {
        py::object numpy = py::module::import("numpy");
        auto global = py::dict();
        global["np"] = numpy;

        m.attr("f") = py::eval("np.vectorize(lambda v: v.f if hasattr(v, 'f') else v)", global);
        m.attr("d") = py::eval("np.vectorize(lambda v: v.g if hasattr(v, 'g') else np.zeros((0)), signature='()->(n)')", global);
        m.attr("dd") = py::eval("np.vectorize(lambda v: v.hm() if hasattr(v, 'hm') else np.zeros((0, 0)), signature='()->(n,m)')", global);

        m.def(
            "variables", [](const std::vector<double> &values, const hj::index order)
            {
                if (order < 0 || 2 < order)
                {
                    throw std::runtime_error("Invalid order");
                }

                py::list results;

                const auto extend = results.attr("extend");

                switch (order)
                {
                case 0:
                    extend(values);
                    break;
                case 1:
                    switch (hj::length(values))
                    {
                    case 0:
                        break;
                    case 1:
                        extend(hj::DDScalar<1, double, 1>::variables(values));
                        break;
                    case 2:
                        extend(hj::DDScalar<1, double, 2>::variables(values));
                        break;
                    case 3:
                        extend(hj::DDScalar<1, double, 3>::variables(values));
                        break;
                    case 4:
                        extend(hj::DDScalar<1, double, 4>::variables(values));
                        break;
                    case 5:
                        extend(hj::DDScalar<1, double, 5>::variables(values));
                        break;
                    case 6:
                        extend(hj::DDScalar<1, double, 6>::variables(values));
                        break;
                    case 7:
                        extend(hj::DDScalar<1, double, 7>::variables(values));
                        break;
                    case 8:
                        extend(hj::DDScalar<1, double, 8>::variables(values));
                        break;
                    case 9:
                        extend(hj::DDScalar<1, double, 9>::variables(values));
                        break;
                    case 10:
                        extend(hj::DDScalar<1, double, 10>::variables(values));
                        break;
                    case 11:
                        extend(hj::DDScalar<1, double, 11>::variables(values));
                        break;
                    case 12:
                        extend(hj::DDScalar<1, double, 12>::variables(values));
                        break;
                    case 13:
                        extend(hj::DDScalar<1, double, 13>::variables(values));
                        break;
                    case 14:
                        extend(hj::DDScalar<1, double, 14>::variables(values));
                        break;
                    case 15:
                        extend(hj::DDScalar<1, double, 15>::variables(values));
                        break;
                    default:
                        extend(hj::DDScalar<1, double, -1>::variables(values));
                        break;
                    }
                    break;
                case 2:
                    switch (hj::length(values))
                    {
                    case 0:
                        break;
                    case 1:
                        extend(hj::DDScalar<2, double, 1>::variables(values));
                        break;
                    case 2:
                        extend(hj::DDScalar<2, double, 2>::variables(values));
                        break;
                    case 3:
                        extend(hj::DDScalar<2, double, 3>::variables(values));
                        break;
                    case 4:
                        extend(hj::DDScalar<2, double, 4>::variables(values));
                        break;
                    case 5:
                        extend(hj::DDScalar<2, double, 5>::variables(values));
                        break;
                    case 6:
                        extend(hj::DDScalar<2, double, 6>::variables(values));
                        break;
                    case 7:
                        extend(hj::DDScalar<2, double, 7>::variables(values));
                        break;
                    case 8:
                        extend(hj::DDScalar<2, double, 8>::variables(values));
                        break;
                    case 9:
                        extend(hj::DDScalar<2, double, 9>::variables(values));
                        break;
                    case 10:
                        extend(hj::DDScalar<2, double, 10>::variables(values));
                        break;
                    case 11:
                        extend(hj::DDScalar<2, double, 11>::variables(values));
                        break;
                    case 12:
                        extend(hj::DDScalar<2, double, 12>::variables(values));
                        break;
                    case 13:
                        extend(hj::DDScalar<2, double, 13>::variables(values));
                        break;
                    case 14:
                        extend(hj::DDScalar<2, double, 14>::variables(values));
                        break;
                    case 15:
                        extend(hj::DDScalar<2, double, 15>::variables(values));
                        break;
                    default:
                        extend(hj::DDScalar<2, double, -1>::variables(values));
                        break;
                    }
                    break;
                }
                return results;
            },
            "values"_a, "order"_a = 2);
    }
}
