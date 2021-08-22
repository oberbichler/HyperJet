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

include(../cmake/CPM.cmake)

# --- Add dependencies

CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.6.0")

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

# --- Create library

add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                            $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

# ---- Create an installable target ----

packageProject(
  NAME ${PROJECT_NAME}
  VERSION ${PROJECT_VERSION}
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)