#include "common.h"

void bind_ddscalar_12d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 12>;
    using DDType = hj::DDScalar<2, double, 12>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 12>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 12>;

    auto d_cls = bind<DType>(m, "D12Scalar");
    auto dd_cls = bind<DDType>(m, "DD12Scalar");
    // auto ds_cls = bind<DSType>(m, "D12SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD12SScalar");
}