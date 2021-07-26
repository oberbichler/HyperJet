#include "common.h"

void bind_ddscalar_9d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<1, double, 9>;
    using DDType = hyperjet::DDScalar<2, double, 9>;

    auto d_cls = bind<DType>(m, "D9Scalar");
    auto dd_cls = bind<DDType>(m, "DD9Scalar");
}