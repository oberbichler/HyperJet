#include "common.h"

void bind_ddscalar_14d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<2, double, 14>;

    auto d_cls = bind<DType>(m, "D14Scalar");
    
    using DDType = hyperjet::DDScalar<2, double, 14>;

    auto dd_cls = bind<DDType>(m, "DD14Scalar");
}