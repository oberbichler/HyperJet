#include "common.h"

void bind_ddscalar_xd(pybind11::module& m)
{
    using DType = hyperjet::DDScalar<2, double>;
    using DDType = hyperjet::DDScalar<2, double>;

    auto d_cls = bind<DType>(m, "DScalar");
    auto dd_cls = bind<DDType>(m, "DDScalar");
}