#include "common.h"

void bind_ddscalar_6d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<2, double, 6>;

    auto d_cls = bind<DType>(m, "D6Scalar");
    
    using DDType = hyperjet::DDScalar<2, double, 6>;

    auto dd_cls = bind<DDType>(m, "DD6Scalar");
}