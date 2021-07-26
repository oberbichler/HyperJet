#include "common.h"

void bind_ddscalar_1d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<2, double, 1>;

    auto d_cls = bind<DType>(m, "D1Scalar");
    
    using DDType = hyperjet::DDScalar<2, double, 1>;

    auto dd_cls = bind<DDType>(m, "DD1Scalar");
}