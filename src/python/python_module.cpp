#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include <hyperjet.h>

template <hyperjet::index TOrder, typename TScalar, hyperjet::index TSize = hyperjet::Dynamic>
void bind_ddscalar(pybind11::module&, const std::string&);

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

    bind_ddscalar<1, double, -1>(m, "DScalar");
    bind_ddscalar<1, double, 1>(m, "D1Scalar");
    bind_ddscalar<1, double, 2>(m, "D2Scalar");
    bind_ddscalar<1, double, 3>(m, "D3Scalar");
    bind_ddscalar<1, double, 4>(m, "D4Scalar");
    bind_ddscalar<1, double, 5>(m, "D5Scalar");
    bind_ddscalar<1, double, 6>(m, "D6Scalar");
    bind_ddscalar<1, double, 7>(m, "D7Scalar");
    bind_ddscalar<1, double, 8>(m, "D8Scalar");
    bind_ddscalar<1, double, 9>(m, "D9Scalar");
    bind_ddscalar<1, double, 10>(m, "D10Scalar");
    bind_ddscalar<1, double, 11>(m, "D11Scalar");
    bind_ddscalar<1, double, 12>(m, "D12Scalar");
    bind_ddscalar<1, double, 13>(m, "D13Scalar");
    bind_ddscalar<1, double, 14>(m, "D14Scalar");
    bind_ddscalar<1, double, 15>(m, "D15Scalar");
    bind_ddscalar<1, double, 16>(m, "D16Scalar");

    bind_ddscalar<2, double>(m, "DDScalar");
    bind_ddscalar<2, double, 1>(m, "DD1Scalar");
    bind_ddscalar<2, double, 2>(m, "DD2Scalar");
    bind_ddscalar<2, double, 3>(m, "DD3Scalar");
    bind_ddscalar<2, double, 4>(m, "DD4Scalar");
    bind_ddscalar<2, double, 5>(m, "DD5Scalar");
    bind_ddscalar<2, double, 6>(m, "DD6Scalar");
    bind_ddscalar<2, double, 7>(m, "DD7Scalar");
    bind_ddscalar<2, double, 8>(m, "DD8Scalar");
    bind_ddscalar<2, double, 9>(m, "DD9Scalar");
    bind_ddscalar<2, double, 10>(m, "DD10Scalar");
    bind_ddscalar<2, double, 11>(m, "DD11Scalar");
    bind_ddscalar<2, double, 12>(m, "DD12Scalar");
    bind_ddscalar<2, double, 13>(m, "DD13Scalar");
    bind_ddscalar<2, double, 14>(m, "DD14Scalar");
    bind_ddscalar<2, double, 15>(m, "DD15Scalar");
    bind_ddscalar<2, double, 16>(m, "DD16Scalar");

    // utilities
    {
        py::object numpy = py::module::import("numpy");
        auto global = py::dict();
        global["np"] = numpy;

        m.attr("f") = py::eval("np.vectorize(lambda v: v.f if hasattr(v, 'f') else v)", global);
        m.attr("d") = py::eval("np.vectorize(lambda v: v.g if hasattr(v, 'g') else np.zeros((0)), signature='()->(n)')", global);
        m.attr("dd") = py::eval("np.vectorize(lambda v: v.hm() if hasattr(v, 'hm') else np.zeros((0, 0)), signature='()->(n,m)')", global);

        m.def("variables", [](const std::vector<double>& values, const hj::index order) {
            if (order < 0 || 2 < order) {
                throw std::runtime_error("Invalid order");
            }

            py::list results;

            const auto extend = results.attr("extend");

            switch (order) {
            case 0:
                extend(values);
                break;
            case 1:
                switch (hj::length(values)) {
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
                switch (hj::length(values)) {
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
        }, "values"_a, "order"_a=2);
    }
}