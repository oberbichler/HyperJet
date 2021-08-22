#include "common.h"

void bind_ddscalar_6d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 6>;
    using DDType = hj::DDScalar<2, double, 6>;
    using DSType = hj::DDScalar<1, hj::SScalar<double>, 6>;
    using DDSType = hj::DDScalar<2, hj::SScalar<double>, 6>;

    auto d_cls = bind<DType>(m, "D6Scalar");
    auto dd_cls = bind<DDType>(m, "DD6Scalar");
    auto ds_cls = bind<DSType>(m, "D6SScalar");
    auto dds_cls = bind<DDSType>(m, "DD6SScalar");
}