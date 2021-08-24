#include "common.h"

void bind_ddscalar_xd(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double>;
    using DDType = hj::DDScalar<2, double>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>>;

    auto d_cls = bind<DType>(m, "DScalar");
    auto dd_cls = bind<DDType>(m, "DDScalar");
    // auto ds_cls = bind<DSType>(m, "DSScalar");
    // auto dds_cls = bind<DDSType>(m, "DDSScalar");
}