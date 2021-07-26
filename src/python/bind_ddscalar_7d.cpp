#include "common.h"

void bind_ddscalar_7d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<1, double, 7>;
    using DDType = hyperjet::DDScalar<2, double, 7>;

    auto d_cls = bind<DType>(m, "D7Scalar");
    auto dd_cls = bind<DDType>(m, "DD7Scalar");
}