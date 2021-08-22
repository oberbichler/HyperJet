#include "common.h"

void bind_ddscalar_1d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 1>;
    using DDType = hj::DDScalar<2, double, 1>;
    using DSType = hj::DDScalar<1, hj::SScalar<double>, 1>;
    using DDSType = hj::DDScalar<2, hj::SScalar<double>, 1>;

    auto d_cls = bind<DType>(m, "D1Scalar");
    auto dd_cls = bind<DDType>(m, "DD1Scalar");
    auto ds_cls = bind<DSType>(m, "D1SScalar");
    auto dds_cls = bind<DDSType>(m, "DD1SScalar");
}