#pragma once

// A NumPy dtype for the statically sized scalars.
//
// Every static variant has a fixed size, so its element size is known at
// compile time and one descriptor per type is enough -- no parametric dtype is
// needed. The dynamic variants hold a std::vector and are neither
// payload-sized nor trivially copyable, so they cannot back a dtype at all.
//
// Requirements NumPy enforces at registration time, none of which are in the
// headers -- they surface as errors from PyArrayInitDTypeMeta_FromSpec:
//
//   * __repr__ and __str__ have to be provided
//   * a cast between the DType's own instances is mandatory
//   * that cast has to handle unaligned data and say so via
//     NPY_METH_SUPPORTS_UNALIGNED
//   * np.dtype.__new__ must not be inherited, the DType needs its own
//
// Arrays have to be asked for explicitly, via np.array(..., dtype=Cls.dtype).
// See the note on spec.typeobj below.
//
// All loops here copy through local values, which makes them safe for
// unaligned data for free.

// The numpy C API is a table of pointers that has to be imported once per
// module. main.cpp defines HYPERJET_IMPORT_ARRAY and performs the import; every
// other unit shares the same table.
#define PY_ARRAY_UNIQUE_SYMBOL hyperjet_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL hyperjet_UFUNC_API

#if !defined(HYPERJET_IMPORT_ARRAY)
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#endif

#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION

#include <pybind11/pybind11.h>

#include <numpy/arrayobject.h>
#include <numpy/dtype_api.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include <cstring>     // memcpy
#include <functional>  // divides, minus, multiplies, plus
#include <string>      // string
#include <type_traits> // is_trivially_copyable

namespace hyperjet_dtype {

namespace py = pybind11;

// One DType, one descriptor and one name per scalar type. Inline variable
// templates give exactly one of each across all translation units.
template <typename T> inline PyArray_DTypeMeta meta{};
template <typename T> inline PyArray_Descr *singleton = nullptr;
template <typename T> inline std::string name{};

template <typename T> PyTypeObject *as_type() {
  return reinterpret_cast<PyTypeObject *>(&meta<T>);
}

// --- descriptor -------------------------------------------------------------

template <typename T> struct Descr {
  PyArray_Descr base;
};

template <typename T> PyArray_Descr *make_descr() {
  auto *d =
      reinterpret_cast<Descr<T> *>(PyArrayDescr_Type.tp_alloc(as_type<T>(), 0));

  if (d == nullptr) {
    return nullptr;
  }

  d->base.elsize = sizeof(T);
  d->base.alignment = alignof(T);
  d->base.flags = NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_PYAPI;
  d->base.type_num = -1;
  d->base.byteorder = '|';
  d->base.kind = 'V';
  d->base.type = 'j';

  return reinterpret_cast<PyArray_Descr *>(d);
}

template <typename T> PyObject *descr_repr(PyObject *) {
  return PyUnicode_FromString(name<T>.c_str());
}

// The size follows from the type, so there is exactly one descriptor.
template <typename T>
PyObject *descr_new(PyTypeObject *, PyObject *args, PyObject *kwargs) {
  if (PyTuple_GET_SIZE(args) != 0 ||
      (kwargs != nullptr && PyDict_Size(kwargs) != 0)) {
    PyErr_Format(PyExc_TypeError, "%s takes no arguments", name<T>.c_str());
    return nullptr;
  }

  Py_INCREF(singleton<T>);

  return reinterpret_cast<PyObject *>(singleton<T>);
}

// --- slots ------------------------------------------------------------------

template <typename T> PyArray_Descr *default_descr(PyArray_DTypeMeta *) {
  Py_INCREF(singleton<T>);

  return singleton<T>;
}

template <typename T>
PyArray_DTypeMeta *common_dtype(PyArray_DTypeMeta *, PyArray_DTypeMeta *) {
  // Scalars of different sizes describe different variable sets, so there is
  // no meaningful common type.
  Py_INCREF(Py_NotImplemented);

  return reinterpret_cast<PyArray_DTypeMeta *>(Py_NotImplemented);
}

template <typename T>
PyArray_Descr *common_instance(PyArray_Descr *a, PyArray_Descr *) {
  Py_INCREF(a);

  return a;
}

template <typename T> PyArray_Descr *ensure_canonical(PyArray_Descr *d) {
  Py_INCREF(d);

  return d;
}

template <typename T> PyObject *getitem(PyArray_Descr *, char *ptr) {
  T value;
  std::memcpy(&value, ptr, sizeof(T));

  try {
    return py::cast(value).release().ptr();
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

template <typename T> int setitem(PyArray_Descr *, PyObject *obj, char *ptr) {
  try {
    const T value = py::reinterpret_borrow<py::object>(obj).template cast<T>();
    std::memcpy(ptr, &value, sizeof(T));
    return 0;
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return -1;
  }
}

// --- loops ------------------------------------------------------------------

template <typename T>
int copy_loop(PyArrayMethod_Context *, char *const *data,
              const npy_intp *dimensions, const npy_intp *strides,
              NpyAuxData *) {
  char *in = data[0];
  char *out = data[1];

  for (npy_intp n = dimensions[0]; n > 0; n--) {
    std::memcpy(out, in, sizeof(T));
    in += strides[0];
    out += strides[1];
  }

  return 0;
}

template <typename T>
NPY_CASTING cast_resolve(PyObject *, PyArray_DTypeMeta *const *,
                         PyArray_Descr *const *given, PyArray_Descr **loop,
                         npy_intp *view_offset) {
  Py_INCREF(given[0]);
  loop[0] = given[0];

  PyArray_Descr *out = given[1] != nullptr ? given[1] : given[0];
  Py_INCREF(out);
  loop[1] = out;

  *view_offset = 0;

  return NPY_NO_CASTING;
}

template <typename T, typename TOp>
int binary_loop(PyArrayMethod_Context *, char *const *data,
                const npy_intp *dimensions, const npy_intp *strides,
                NpyAuxData *) {
  char *a = data[0];
  char *b = data[1];
  char *r = data[2];

  for (npy_intp n = dimensions[0]; n > 0; n--) {
    T va;
    T vb;
    std::memcpy(&va, a, sizeof(T));
    std::memcpy(&vb, b, sizeof(T));

    const T vr = TOp{}(va, vb);
    std::memcpy(r, &vr, sizeof(T));

    a += strides[0];
    b += strides[1];
    r += strides[2];
  }

  return 0;
}

template <typename T, T (T::*TFn)() const>
int unary_method_loop(PyArrayMethod_Context *, char *const *data,
                      const npy_intp *dimensions, const npy_intp *strides,
                      NpyAuxData *) {
  char *a = data[0];
  char *r = data[1];

  for (npy_intp n = dimensions[0]; n > 0; n--) {
    T va;
    std::memcpy(&va, a, sizeof(T));

    const T vr = (va.*TFn)();
    std::memcpy(r, &vr, sizeof(T));

    a += strides[0];
    r += strides[1];
  }

  return 0;
}

// atan2 is a member, hypot a static, so neither fits std::plus and friends.
template <typename T> struct Atan2Op {
  T operator()(const T &a, const T &b) const { return a.atan2(b); }
};

template <typename T> struct HypotOp {
  T operator()(const T &a, const T &b) const { return T::hypot(a, b); }
};

template <typename T> struct PositiveOp {
  T operator()(const T &a) const { return a; }
};

template <typename T, typename TOp>
int unary_op_loop(PyArrayMethod_Context *, char *const *data,
                  const npy_intp *dimensions, const npy_intp *strides,
                  NpyAuxData *) {
  char *a = data[0];
  char *r = data[1];

  for (npy_intp n = dimensions[0]; n > 0; n--) {
    T va;
    std::memcpy(&va, a, sizeof(T));

    const T vr = TOp{}(va);
    std::memcpy(r, &vr, sizeof(T));

    a += strides[0];
    r += strides[1];
  }

  return 0;
}

template <typename T>
int negative_loop(PyArrayMethod_Context *, char *const *data,
                  const npy_intp *dimensions, const npy_intp *strides,
                  NpyAuxData *) {
  char *a = data[0];
  char *r = data[1];

  for (npy_intp n = dimensions[0]; n > 0; n--) {
    T va;
    std::memcpy(&va, a, sizeof(T));

    const T vr = -va;
    std::memcpy(r, &vr, sizeof(T));

    a += strides[0];
    r += strides[1];
  }

  return 0;
}

// np.matmul and np.vecdot are generalized ufuncs, so they take loops. Their
// dimensions and strides carry the core axes after the outer ones.
template <typename T>
int vecdot_loop(PyArrayMethod_Context *, char *const *data,
                const npy_intp *dimensions, const npy_intp *strides,
                NpyAuxData *) {
  char *a = data[0];
  char *b = data[1];
  char *r = data[2];

  for (npy_intp k = dimensions[0]; k > 0; k--) {
    const char *pa = a;
    const char *pb = b;

    T acc = T::zero();

    for (npy_intp i = dimensions[1]; i > 0; i--) {
      T va;
      T vb;
      std::memcpy(&va, pa, sizeof(T));
      std::memcpy(&vb, pb, sizeof(T));

      acc += va * vb;

      pa += strides[3];
      pb += strides[4];
    }

    std::memcpy(r, &acc, sizeof(T));

    a += strides[0];
    b += strides[1];
    r += strides[2];
  }

  return 0;
}

template <typename T>
int matmul_loop(PyArrayMethod_Context *, char *const *data,
                const npy_intp *dimensions, const npy_intp *strides,
                NpyAuxData *) {
  char *a = data[0];
  char *b = data[1];
  char *r = data[2];

  for (npy_intp o = dimensions[0]; o > 0; o--) {
    for (npy_intp i = 0; i < dimensions[1]; i++) {
      for (npy_intp j = 0; j < dimensions[3]; j++) {
        T acc = T::zero();

        for (npy_intp k = 0; k < dimensions[2]; k++) {
          T va;
          T vb;
          std::memcpy(&va, a + i * strides[3] + k * strides[4], sizeof(T));
          std::memcpy(&vb, b + k * strides[5] + j * strides[6], sizeof(T));

          acc += va * vb;
        }

        std::memcpy(r + i * strides[7] + j * strides[8], &acc, sizeof(T));
      }
    }

    a += strides[0];
    b += strides[1];
    r += strides[2];
  }

  return 0;
}

template <typename T, int TArity>
NPY_CASTING ufunc_resolve(PyObject *, PyArray_DTypeMeta *const *,
                          PyArray_Descr *const *given, PyArray_Descr **loop,
                          npy_intp *) {
  for (int i = 0; i < TArity + 1; i++) {
    PyArray_Descr *d = i < TArity ? given[i] : nullptr;

    if (d == nullptr) {
      d = singleton<T>;
    }

    Py_INCREF(d);
    loop[i] = d;
  }

  return NPY_NO_CASTING;
}

// --- registration -----------------------------------------------------------

inline bool add_loop(const char *ufunc_name, PyArrayMethod_Spec *spec) {
  py::object ufunc = py::module::import("numpy").attr(ufunc_name);

  return PyUFunc_AddLoopFromSpec(ufunc.ptr(), spec) >= 0;
}

template <typename T, int TArity>
bool register_loop(const char *ufunc_name, const char *loop_name,
                   PyArrayMethod_StridedLoop *loop) {
  // NumPy may keep pointers into the spec, and every registration needs its
  // own. Function-local statics would be shared by all calls with the same
  // <T, TArity>, so these live on the heap for the lifetime of the module.
  auto *dtypes = new PyArray_DTypeMeta *[TArity + 1];
  auto *slots = new PyType_Slot[3];
  auto *spec_storage = new PyArrayMethod_Spec{};
  auto &spec = *spec_storage;

  for (int i = 0; i < TArity + 1; i++) {
    dtypes[i] = &meta<T>;
  }

  slots[0] = {NPY_METH_strided_loop, reinterpret_cast<void *>(loop)};
  slots[1] = {NPY_METH_resolve_descriptors,
              reinterpret_cast<void *>(&ufunc_resolve<T, TArity>)};
  slots[2] = {0, nullptr};

  spec.name = loop_name;
  spec.nin = TArity;
  spec.nout = 1;
  spec.casting = NPY_NO_CASTING;
  spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
  spec.dtypes = dtypes;
  spec.slots = slots;

  return add_loop(ufunc_name, &spec);
}

// Registers the dtype for a statically sized scalar and its arithmetic loops.
template <typename T> void bind_dtype(py::module &m, py::object scalar_class) {
  static_assert(!T::is_dynamic(),
                "only statically sized scalars have a fixed element size");
  static_assert(std::is_trivially_copyable_v<T>,
                "elements are moved around with memcpy");
  static_assert(sizeof(T) == sizeof(typename T::Data),
                "the element must be exactly its data");

  name<T> = py::cast<std::string>(scalar_class.attr("__name__")) + "DType";

  auto *type = as_type<T>();
  Py_SET_REFCNT(type, 1);
  Py_SET_TYPE(type, &PyArrayDTypeMeta_Type);
  type->tp_base = &PyArrayDescr_Type;
  type->tp_name = name<T>.c_str();
  type->tp_basicsize = sizeof(Descr<T>);
  type->tp_flags = Py_TPFLAGS_DEFAULT;
  type->tp_repr = &descr_repr<T>;
  type->tp_str = &descr_repr<T>;
  type->tp_new = &descr_new<T>;
  meta<T>.scalar_type = nullptr;

  if (PyType_Ready(type) < 0) {
    throw py::error_already_set();
  }

  static PyType_Slot slots[] = {
      {NPY_DT_default_descr, reinterpret_cast<void *>(&default_descr<T>)},
      {NPY_DT_common_dtype, reinterpret_cast<void *>(&common_dtype<T>)},
      {NPY_DT_common_instance, reinterpret_cast<void *>(&common_instance<T>)},
      {NPY_DT_ensure_canonical, reinterpret_cast<void *>(&ensure_canonical<T>)},
      {NPY_DT_setitem, reinterpret_cast<void *>(&setitem<T>)},
      {NPY_DT_getitem, reinterpret_cast<void *>(&getitem<T>)},
      {0, nullptr}};

  // NumPy insists on a cast between the DType's own instances, and that it
  // handles unaligned data.
  static PyArray_DTypeMeta *cast_dtypes[] = {nullptr, nullptr};
  static PyType_Slot cast_slots[] = {
      {NPY_METH_strided_loop, reinterpret_cast<void *>(&copy_loop<T>)},
      {NPY_METH_unaligned_strided_loop,
       reinterpret_cast<void *>(&copy_loop<T>)},
      {NPY_METH_resolve_descriptors,
       reinterpret_cast<void *>(&cast_resolve<T>)},
      {0, nullptr}};
  static PyArrayMethod_Spec cast_spec = {};
  cast_spec.name = "hyperjet_copy";
  cast_spec.nin = 1;
  cast_spec.nout = 1;
  cast_spec.casting = NPY_NO_CASTING;
  cast_spec.flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(
      NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_SUPPORTS_UNALIGNED);
  cast_spec.dtypes = cast_dtypes;
  cast_spec.slots = cast_slots;
  static PyArrayMethod_Spec *casts[] = {&cast_spec, nullptr};

  PyArrayDTypeMeta_Spec spec = {};

  // Deliberately not the scalar class. Associating the two would make
  // np.array([x, y, z]) pick this dtype automatically, and every ufunc without
  // a loop yet -- sin, sqrt, abs and the rest -- would then fail instead of
  // falling back to the object path. Measured: 33 failing tests. The
  // association belongs in the commit that completes the loop set. NumPy does
  // not accept a null type object, so this stands in until then.
  spec.typeobj = reinterpret_cast<PyTypeObject *>(scalar_class.ptr());
  spec.flags = 0;
  spec.casts = casts;
  spec.slots = slots;
  spec.baseclass = nullptr;

  if (PyArrayInitDTypeMeta_FromSpec(&meta<T>, &spec) < 0) {
    throw py::error_already_set();
  }

  singleton<T> = make_descr<T>();

  if (singleton<T> == nullptr) {
    throw py::error_already_set();
  }

#define HYPERJET_UNARY(ufunc, method)                                          \
  register_loop<T, 1>(ufunc, "hyperjet_" ufunc,                                \
                      &unary_method_loop<T, &T::method>)

  const bool ok =
      register_loop<T, 2>("add", "hyperjet_add",
                          &binary_loop<T, std::plus<T>>) &&
      register_loop<T, 2>("subtract", "hyperjet_subtract",
                          &binary_loop<T, std::minus<T>>) &&
      register_loop<T, 2>("multiply", "hyperjet_multiply",
                          &binary_loop<T, std::multiplies<T>>) &&
      register_loop<T, 2>("divide", "hyperjet_divide",
                          &binary_loop<T, std::divides<T>>) &&
      register_loop<T, 2>("arctan2", "hyperjet_arctan2",
                          &binary_loop<T, Atan2Op<T>>) &&
      register_loop<T, 2>("hypot", "hyperjet_hypot",
                          &binary_loop<T, HypotOp<T>>) &&
      register_loop<T, 1>("negative", "hyperjet_negative", &negative_loop<T>) &&
      register_loop<T, 1>("positive", "hyperjet_positive",
                          &unary_op_loop<T, PositiveOp<T>>) &&
      HYPERJET_UNARY("absolute", abs) &&
      HYPERJET_UNARY("reciprocal", reciprocal) &&
      HYPERJET_UNARY("sqrt", sqrt) && HYPERJET_UNARY("cbrt", cbrt) &&
      HYPERJET_UNARY("sin", sin) && HYPERJET_UNARY("cos", cos) &&
      HYPERJET_UNARY("tan", tan) && HYPERJET_UNARY("arcsin", asin) &&
      HYPERJET_UNARY("arccos", acos) && HYPERJET_UNARY("arctan", atan) &&
      HYPERJET_UNARY("sinh", sinh) && HYPERJET_UNARY("cosh", cosh) &&
      HYPERJET_UNARY("tanh", tanh) && HYPERJET_UNARY("arcsinh", asinh) &&
      HYPERJET_UNARY("arccosh", acosh) && HYPERJET_UNARY("arctanh", atanh) &&
      HYPERJET_UNARY("exp", exp) && HYPERJET_UNARY("log2", log2) &&
      HYPERJET_UNARY("log10", log10) &&
      register_loop<T, 1>(
          "log", "hyperjet_log",
          &unary_method_loop<T, static_cast<T (T::*)() const>(&T::log)>) &&
      register_loop<T, 2>("vecdot", "hyperjet_vecdot", &vecdot_loop<T>) &&
      register_loop<T, 2>("matmul", "hyperjet_matmul", &matmul_loop<T>);

#undef HYPERJET_UNARY

  if (!ok) {
    throw py::error_already_set();
  }

  // reachable as e.g. hj.DD3Scalar.dtype
  scalar_class.attr("dtype") = py::reinterpret_borrow<py::object>(
      reinterpret_cast<PyObject *>(singleton<T>));

  m.attr(name<T>.c_str()) = py::reinterpret_borrow<py::object>(
      reinterpret_cast<PyObject *>(&meta<T>));
}

// A non-template `if constexpr` would still instantiate the discarded call,
// so the guard lives in a template of its own.
template <typename T>
void bind_dtype_if_static(py::module &m, py::object scalar_class) {
  if constexpr (!T::is_dynamic()) {
    bind_dtype<T>(m, scalar_class);
  }
}

} // namespace hyperjet_dtype
