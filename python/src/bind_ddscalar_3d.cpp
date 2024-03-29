#include "common.h"

void bind_ddscalar_3d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 3>;
    using DDType = hj::DDScalar<2, double, 3>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 3>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 3>;

    auto d_cls = bind<DType>(m, "D3Scalar");
    auto dd_cls = bind<DDType>(m, "DD3Scalar");
    // auto ds_cls = bind<DSType>(m, "D3SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD3SScalar");
}