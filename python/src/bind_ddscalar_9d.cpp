#include "common.h"

void bind_ddscalar_9d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 9>;
    using DDType = hj::DDScalar<2, double, 9>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 9>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 9>;

    auto d_cls = bind<DType>(m, "D9Scalar");
    auto dd_cls = bind<DDType>(m, "DD9Scalar");
    // auto ds_cls = bind<DSType>(m, "D9SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD9SScalar");
}