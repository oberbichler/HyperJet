#include "common.h"

void bind_ddscalar_5d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 5>;
    using DDType = hj::DDScalar<2, double, 5>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 5>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 5>;

    auto d_cls = bind<DType>(m, "D5Scalar");
    auto dd_cls = bind<DDType>(m, "DD5Scalar");
    // auto ds_cls = bind<DSType>(m, "D5SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD5SScalar");
}