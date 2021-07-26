#include "common.h"

void bind_ddscalar_0d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<1, double, 0>;
    using DDType = hyperjet::DDScalar<2, double, 0>;

    auto d_cls = bind<DType>(m, "D0Scalar");
    auto dd_cls = bind<DDType>(m, "DD0Scalar");
}