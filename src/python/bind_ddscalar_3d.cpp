#include "common.h"

void bind_ddscalar_3d(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<1, double, 3>;
    using DDType = hyperjet::DDScalar<2, double, 3>;

    auto d_cls = bind<DType>(m, "D3Scalar");
    auto dd_cls = bind<DDType>(m, "DD3Scalar");
}