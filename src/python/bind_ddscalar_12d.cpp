#include "common.h"

void bind_ddscalar_12d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<2, double, 12>;

    auto d_cls = bind<DType>(m, "D12Scalar");
    
    using DDType = hyperjet::DDScalar<2, double, 12>;

    auto dd_cls = bind<DDType>(m, "DD12Scalar");
}