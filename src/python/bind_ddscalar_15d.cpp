#include "common.h"

void bind_ddscalar_15d(pybind11::module &m)
{
    using DType = hyperjet::DDScalar<1, double, 15>;
    using DDType = hyperjet::DDScalar<2, double, 15>;

    auto d_cls = bind<DType>(m, "D15Scalar");
    auto dd_cls = bind<DDType>(m, "DD15Scalar");
}