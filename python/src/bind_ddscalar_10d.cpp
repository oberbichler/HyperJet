#include "common.h"

void bind_ddscalar_10d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 10>;
    using DDType = hj::DDScalar<2, double, 10>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 10>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 10>;

    auto d_cls = bind<DType>(m, "D10Scalar");
    auto dd_cls = bind<DDType>(m, "DD10Scalar");
    // auto ds_cls = bind<DSType>(m, "D10SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD10SScalar");
}