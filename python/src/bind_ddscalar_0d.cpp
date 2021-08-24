#include "common.h"

void bind_ddscalar_0d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 0>;
    using DDType = hj::DDScalar<2, double, 0>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 0>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 0>;

    auto d_cls = bind<DType>(m, "D0Scalar");
    auto dd_cls = bind<DDType>(m, "DD0Scalar");
    // auto ds_cls = bind<DSType>(m, "D0SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD0SScalar");
}