#include "common.h"

void bind_ddscalar_5d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<1, double, 5>;
    using DDType = hyperjet::DDScalar<2, double, 5>;

    auto d_cls = bind<DType>(m, "D5Scalar");
    auto dd_cls = bind<DDType>(m, "DD5Scalar");
}