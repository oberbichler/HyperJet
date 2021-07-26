#include "common.h"

void bind_ddscalar_8d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<2, double, 8>;

    auto d_cls = bind<DType>(m, "D8Scalar");
    
    using DDType = hyperjet::DDScalar<2, double, 8>;

    auto dd_cls = bind<DDType>(m, "DD8Scalar");
}