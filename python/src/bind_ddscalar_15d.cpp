#include "common.h"

void bind_ddscalar_15d(pybind11::module &m)
{
    using DType = hj::DDScalar<1, double, 15>;
    using DDType = hj::DDScalar<2, double, 15>;
    // using DSType = hj::DDScalar<1, hj::SScalar<double>, 15>;
    // using DDSType = hj::DDScalar<2, hj::SScalar<double>, 15>;

    auto d_cls = bind<DType>(m, "D15Scalar");
    auto dd_cls = bind<DDType>(m, "DD15Scalar");
    // auto ds_cls = bind<DSType>(m, "D15SScalar");
    // auto dds_cls = bind<DDSType>(m, "DD15SScalar");
}