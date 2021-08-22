#include "common.h"

void bind_ddscalar_2d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 2>;
    using DDType = hj::DDScalar<2, double, 2>;
    using DSType = hj::DDScalar<1, hj::SScalar<double>, 2>;
    using DDSType = hj::DDScalar<2, hj::SScalar<double>, 2>;

    auto d_cls = bind<DType>(m, "D2Scalar");
    auto dd_cls = bind<DDType>(m, "DD2Scalar");
    auto ds_cls = bind<DSType>(m, "D2SScalar");
    auto dds_cls = bind<DDSType>(m, "DD2SScalar");
}