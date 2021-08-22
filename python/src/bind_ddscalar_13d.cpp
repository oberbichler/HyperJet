#include "common.h"

void bind_ddscalar_13d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 13>;
    using DDType = hj::DDScalar<2, double, 13>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 13>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 13>;

    auto d_cls = bind<DType>(m, "D13Scalar");
    auto dd_cls = bind<DDType>(m, "DD13Scalar");
    // auto ds_cls = bind<DSType>(m, "D13SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD13SScalar");
}