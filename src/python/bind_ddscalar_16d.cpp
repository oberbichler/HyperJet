#include "common.h"

void bind_ddscalar_16d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<2, double, 16>;

    auto d_cls = bind<DType>(m, "D16Scalar");
    
    using DDType = hyperjet::DDScalar<2, double, 16>;

    auto dd_cls = bind<DDType>(m, "DD16Scalar");
}