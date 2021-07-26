#include "common.h"

void bind_ddscalar_12d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<1, double, 12>;
    using DDType = hyperjet::DDScalar<2, double, 12>;

    auto d_cls = bind<DType>(m, "D12Scalar");
    auto dd_cls = bind<DDType>(m, "DD12Scalar");
}