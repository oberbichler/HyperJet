#include "common.h"

void bind_ddscalar_4d(pybind11::module &m)
{
    using DType = hyperjet::DDScalar<1, double, 4>;
    using DDType = hyperjet::DDScalar<2, double, 4>;

    auto d_cls = bind<DType>(m, "D4Scalar");
    auto dd_cls = bind<DDType>(m, "DD4Scalar");
}