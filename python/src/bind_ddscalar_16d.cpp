#include "common.h"

void bind_ddscalar_16d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 16>;
    using DDType = hj::DDScalar<2, double, 16>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 16>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 16>;

    auto d_cls = bind<DType>(m, "D16Scalar");
    auto dd_cls = bind<DDType>(m, "DD16Scalar");
    // auto ds_cls = bind<DSType>(m, "D16SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD16SScalar");
}