#include "common.h"

void bind_ddscalar_2d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<1, double, 2>;
    using DDType = hyperjet::DDScalar<2, double, 2>;

    auto d_cls = bind<DType>(m, "D2Scalar");
    auto dd_cls = bind<DDType>(m, "DD2Scalar");
}