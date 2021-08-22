#include "common.h"

void bind_ddscalar_4d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 4>;
    using DDType = hj::DDScalar<2, double, 4>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 4>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 4>;

    auto d_cls = bind<DType>(m, "D4Scalar");
    auto dd_cls = bind<DDType>(m, "DD4Scalar");
    // auto ds_cls = bind<DSType>(m, "D4SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD4SScalar");
}