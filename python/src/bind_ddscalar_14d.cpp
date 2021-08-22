#include "common.h"

void bind_ddscalar_14d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 14>;
    using DDType = hj::DDScalar<2, double, 14>;
    using DSType = hj::DDScalar<1, hj::SScalar<double>, 14>;
    using DDSType = hj::DDScalar<2, hj::SScalar<double>, 14>;

    auto d_cls = bind<DType>(m, "D14Scalar");
    auto dd_cls = bind<DDType>(m, "DD14Scalar");
    auto ds_cls = bind<DSType>(m, "D14SScalar");
    auto dds_cls = bind<DDSType>(m, "DD14SScalar");
}