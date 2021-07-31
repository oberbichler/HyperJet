#include "common.h"

void bind_ddscalar_14d(pybind11::module &m)
{
    using DType = hyperjet::DDScalar<1, double, 14>;
    using DDType = hyperjet::DDScalar<2, double, 14>;

    auto d_cls = bind<DType>(m, "D14Scalar");
    auto dd_cls = bind<DDType>(m, "DD14Scalar");
}