#include "common.h"

void bind_ddscalar_11d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<1, double, 11>;
    using DDType = hyperjet::DDScalar<2, double, 11>;

    auto d_cls = bind<DType>(m, "D11Scalar");
    auto dd_cls = bind<DDType>(m, "DD11Scalar");
}