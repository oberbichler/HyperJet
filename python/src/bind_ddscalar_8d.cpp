#include "common.h"

void bind_ddscalar_8d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 8>;
    using DDType = hj::DDScalar<2, double, 8>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 8>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 8>;

    auto d_cls = bind<DType>(m, "D8Scalar");
    auto dd_cls = bind<DDType>(m, "DD8Scalar");
    // auto ds_cls = bind<DSType>(m, "D8SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD8SScalar");
}