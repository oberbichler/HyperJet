cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(HyperJetPython LANGUAGES CXX)

# --- Dependencies

include(../cmake/CPM.cmake)

CPMAddPackage(
  NAME Eigen
  VERSION 3.4.0
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_ONLY YES 
)

if(Eigen_ADDED)
  add_library(Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()

# pybind11 and numpy are provided by scikit-build-core build requirements
find_package(pybind11 CONFIG REQUIRED)

# the dtype is written against the numpy C API, so its headers are needed
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE numpy_include
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

if(TEST_INSTALLED_VERSION)
  find_package(HyperJet REQUIRED)
else()
  CPMAddPackage(NAME HyperJet SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()

# --- Generate the DDScalar bindings
#
# The static sizes differ only in a number, so the translation units are
# generated from a template. This is the single place the list of sizes exists:
# the bindings, the class names and the dispatch in variables() all follow from
# it, and they cannot drift apart.

set(HYPERJET_MAX_STATIC_SIZE 16 CACHE STRING "Largest statically sized variant")

set(generated_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(generated_sources)
set(HJ_DECLS "")
set(HJ_CALLS "")
set(HJ_CASES "")

function(hyperjet_add_variant tag size name)
  set(HJ_TAG ${tag})
  set(HJ_SIZE ${size})
  set(HJ_NAME ${name})

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bind_ddscalar.cpp.in
    ${generated_dir}/bind_ddscalar_${tag}.cpp @ONLY
  )

  set(generated_sources ${generated_sources} ${generated_dir}/bind_ddscalar_${tag}.cpp PARENT_SCOPE)
  set(HJ_DECLS
      "${HJ_DECLS}void bind_ddscalar_${tag}(pybind11::module &);\nvoid variables_${tag}(pybind11::list &, const std::vector<double> &, const hj::index);\n"
      PARENT_SCOPE
  )
  set(HJ_CALLS "${HJ_CALLS}  bind_ddscalar_${tag}(m);\n" PARENT_SCOPE)
endfunction()

foreach(size RANGE 0 ${HYPERJET_MAX_STATIC_SIZE})
  hyperjet_add_variant(${size}d ${size} ${size})
  set(HJ_CASES "${HJ_CASES}  case ${size}:\n    variables_${size}d(result, values, order);\n    break;\n")
endforeach()

# The dynamic variant carries no number in its name and is the fallback of the
# dispatch, so it gets no case of its own.
hyperjet_add_variant(xd hj::Dynamic "")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/bind_ddscalar_all.cpp.in
  ${generated_dir}/bind_ddscalar_all.cpp @ONLY
)
list(APPEND generated_sources ${generated_dir}/bind_ddscalar_all.cpp)

# --- Create binary

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
pybind11_add_module(HyperJetPython ${sources} ${generated_sources})

# the generated units include "common.h" from src/
target_include_directories(
  HyperJetPython PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(HyperJetPython SYSTEM PRIVATE ${numpy_include})

target_link_libraries(HyperJetPython PRIVATE hyperjet::hyperjet Eigen)

target_compile_definitions(HyperJetPython PRIVATE
  -DHYPERJET_VERSION="${HYPERJET_VERSION}"
  -DHYPERJET_EXCEPTIONS
)

set_target_properties(HyperJetPython PROPERTIES CXX_STANDARD 23)

set_target_properties(HyperJetPython PROPERTIES OUTPUT_NAME hyperjet)

install(TARGETS HyperJetPython DESTINATION .)
