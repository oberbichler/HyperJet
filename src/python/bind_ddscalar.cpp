#include "common.h"

#include <hyperjet.h>

template <hyperjet::index TOrder, typename TScalar, hyperjet::index TSize = hyperjet::Dynamic>
void bind_ddscalar(pybind11::module& m, const std::string& name)
{
    using Type = hyperjet::DDScalar<TOrder, TScalar, TSize>;

    auto cls = bind<Type>(m, name);

    m.def("hypot", py::overload_cast<const Type&, const Type&>(&Type::hypot));
    m.def("hypot", py::overload_cast<const Type&, const Type&, const Type&>(&Type::hypot));
}

template void bind_ddscalar<1, double>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 1>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 2>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 3>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 4>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 5>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 6>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 7>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 8>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 9>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 10>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 11>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 12>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 13>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 14>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 15>(pybind11::module&, const std::string&);
template void bind_ddscalar<1, double, 16>(pybind11::module&, const std::string&);

template void bind_ddscalar<2, double>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 1>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 2>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 3>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 4>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 5>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 6>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 7>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 8>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 9>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 10>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 11>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 12>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 13>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 14>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 15>(pybind11::module&, const std::string&);
template void bind_ddscalar<2, double, 16>(pybind11::module&, const std::string&);