#include "common.h"

void bind_ddscalar_10d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<2, double, 10>;

    auto d_cls = bind<DType>(m, "D10Scalar");
    
    using DDType = hyperjet::DDScalar<2, double, 10>;

    auto dd_cls = bind<DDType>(m, "DD10Scalar");
}