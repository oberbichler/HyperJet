cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# --- Project

project(
  hyperjet
  VERSION 2.0.0
  LANGUAGES CXX
)

# ---- Include guards

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# --- Get CPM

set(CPM_DOWNLOAD_VERSION 0.32.2)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})

# --- Add dependencies

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.5.6
  OPTIONS
   "BENCHMARK_ENABLE_TESTING Off"
)

if (benchmark_ADDED)
  set_target_properties(benchmark PROPERTIES CXX_STANDARD 17)
endif()

CPMAddPackage("gh:/catchorg/Catch2@2.13.7")

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

CPMAddPackage("gh:/pybind/pybind11@2.7.1")

# ---

option(BUILD_PYTHON_MODULE "Build the Python module" ON )
option(BUILD_TESTS         "Build the C++ tests"     ON )
option(BUILD_BENCHMARKS    "Build the benchmarks"    OFF)

include_directories("include")

# --- tests

if (${BUILD_TESTS})
    add_executable(tests src/test/tests.cpp)

    target_compile_features(tests PRIVATE cxx_std_17)

    target_link_libraries(tests PRIVATE Catch2::Catch2)
    target_link_libraries(tests PRIVATE Eigen)

    install(TARGETS tests DESTINATION bin)
endif()

# --- benchmarks

if (${BUILD_BENCHMARKS})
    add_executable(benchmarks src/benchmark/benchmarks.cpp)

    target_compile_features(benchmarks PRIVATE cxx_std_17)

    target_link_libraries(benchmarks PRIVATE benchmark::benchmark)
    target_link_libraries(benchmarks PRIVATE Eigen)

    install(TARGETS benchmarks DESTINATION bin)
endif()

# --- python_module

if (${BUILD_PYTHON_MODULE})
    include_directories("external_libraries")

    pybind11_add_module(python_module src/python/main.cpp
      src/python/bind_ddscalar_xd.cpp
      src/python/bind_ddscalar_0d.cpp
      src/python/bind_ddscalar_1d.cpp
      src/python/bind_ddscalar_2d.cpp
      src/python/bind_ddscalar_3d.cpp
      src/python/bind_ddscalar_4d.cpp
      src/python/bind_ddscalar_5d.cpp
      src/python/bind_ddscalar_6d.cpp
      src/python/bind_ddscalar_7d.cpp
      src/python/bind_ddscalar_8d.cpp
      src/python/bind_ddscalar_9d.cpp
      src/python/bind_ddscalar_10d.cpp
      src/python/bind_ddscalar_11d.cpp
      src/python/bind_ddscalar_12d.cpp
      src/python/bind_ddscalar_13d.cpp
      src/python/bind_ddscalar_14d.cpp
      src/python/bind_ddscalar_15d.cpp
      src/python/bind_ddscalar_16d.cpp
      src/python/bind_sscalar.cpp
      )

    target_link_libraries(python_module PRIVATE Eigen)

    target_compile_definitions(python_module PRIVATE
        -DEIGEN_DEFAULT_TO_ROW_MAJOR
        -DHYPERJET_VERSION="${HYPERJET_VERSION}"
        -DHYPERJET_EXCEPTIONS
    )

    target_compile_features(python_module PRIVATE cxx_std_17)

    set_target_properties(python_module PROPERTIES OUTPUT_NAME hyperjet)

    install(TARGETS python_module DESTINATION bin)
endif()
