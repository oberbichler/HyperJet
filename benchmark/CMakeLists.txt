cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(Benchmark LANGUAGES CXX)

# --- Dependencies

include(../cmake/CPM.cmake)

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.9.1
  OPTIONS
   "BENCHMARK_ENABLE_TESTING Off"
)

if (benchmark_ADDED)
  set_target_properties(benchmark PROPERTIES CXX_STANDARD 23)
endif()

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

CPMAddPackage("gh:/pybind/pybind11@2.13.6")

# For the comparison benchmark only: another forward-mode AD library to measure
# against. Header-only, so DOWNLOAD_ONLY is enough.
CPMAddPackage(
  NAME autodiff
  GITHUB_REPOSITORY autodiff/autodiff
  GIT_TAG v1.1.2
  DOWNLOAD_ONLY YES
)

# Ceres for the first-order comparison. ceres::Jet is first order only, and
# jet.h needs nothing but its own include tree and Eigen -- no glog, no Abseil,
# so the whole solver does not have to be built.
CPMAddPackage(
  NAME ceres
  GITHUB_REPOSITORY ceres-solver/ceres-solver
  GIT_TAG 2.2.0
  DOWNLOAD_ONLY YES
)

if(ceres_ADDED)
  add_library(ceres_jet INTERFACE IMPORTED)
  target_include_directories(ceres_jet INTERFACE ${ceres_SOURCE_DIR}/include)
endif()

if(autodiff_ADDED)
  add_library(autodiff INTERFACE IMPORTED)
  target_include_directories(autodiff INTERFACE ${autodiff_SOURCE_DIR})
  # autodiff gates its Eigen overloads on this rather than detecting Eigen
  target_compile_definitions(autodiff INTERFACE AUTODIFF_EIGEN_FOUND)
endif()

if(TEST_INSTALLED_VERSION)
  find_package(HyperJet REQUIRED)
else()
  CPMAddPackage(NAME HyperJet SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()

# --- Create binary

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(Benchmark ${sources})

target_link_libraries(Benchmark PRIVATE benchmark::benchmark Eigen hyperjet::hyperjet)

target_compile_definitions(Benchmark PRIVATE
  -DHYPERJET_VERSION="${PROJECT_VERSION}"
  -DHYPERJET_EXCEPTIONS
)

set_target_properties(Benchmark PROPERTIES CXX_STANDARD 23)

# --- Comparison benchmark
#
# A separate target: it pulls in autodiff, which the micro-benchmarks have no
# use for, and it has its own main() that verifies the contenders agree before
# timing anything.

add_executable(Compare ${CMAKE_CURRENT_SOURCE_DIR}/compare/compare.cpp)

target_link_libraries(Compare PRIVATE benchmark::benchmark Eigen autodiff
  ceres_jet hyperjet::hyperjet)

target_compile_definitions(Compare PRIVATE -DHYPERJET_EXCEPTIONS)

set_target_properties(Compare PROPERTIES CXX_STANDARD 23)
