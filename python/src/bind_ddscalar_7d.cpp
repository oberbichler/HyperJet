#include "common.h"

void bind_ddscalar_7d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 7>;
    using DDType = hj::DDScalar<2, double, 7>;
    using DSType = hj::DDScalar<1, hj::SScalar<double>, 7>;
    using DDSType = hj::DDScalar<2, hj::SScalar<double>, 7>;

    auto d_cls = bind<DType>(m, "D7Scalar");
    auto dd_cls = bind<DDType>(m, "DD7Scalar");
    auto ds_cls = bind<DSType>(m, "D7SScalar");
    auto dds_cls = bind<DDSType>(m, "DD7SScalar");
}