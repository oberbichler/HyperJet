#include "common.h"

void bind_ddscalar_11d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 11>;
    using DDType = hj::DDScalar<2, double, 11>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 11>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 11>;

    auto d_cls = bind<DType>(m, "D11Scalar");
    auto dd_cls = bind<DDType>(m, "DD11Scalar");
    // auto ds_cls = bind<DSType>(m, "D11SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD11SScalar");
}