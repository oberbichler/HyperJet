#include "common.h"

void bind_ddscalar_13d(pybind11::module &m)
{
    using DType = hyperjet::DDScalar<1, double, 13>;
    using DDType = hyperjet::DDScalar<2, double, 13>;

    auto d_cls = bind<DType>(m, "D13Scalar");
    auto dd_cls = bind<DDType>(m, "DD13Scalar");
}