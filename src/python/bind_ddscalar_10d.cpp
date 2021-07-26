#include "common.h"

void bind_ddscalar_10d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<1, double, 10>;
    using DDType = hyperjet::DDScalar<2, double, 10>;

    auto d_cls = bind<DType>(m, "D10Scalar");
    auto dd_cls = bind<DDType>(m, "DD10Scalar");
}