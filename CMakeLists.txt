cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

project(hyperjet)

option(BUILD_PYTHON_MODULE "Build the Python module" ON )
option(BUILD_TESTS         "Build the C++ tests"     ON )
option(BUILD_BENCHMARKS    "Build the benchmarks"    OFF)

include_directories(
    "include"
    "external_libraries"
)

# --- tests

if (${BUILD_TESTS})
    add_executable (tests src/tests.cpp)

    install (TARGETS tests DESTINATION bin)
endif ()

# --- benchmarks

if (${BUILD_BENCHMARKS})
    find_package (benchmark REQUIRED)

    add_executable (benchmarks src/benchmarks.cpp)

    target_link_libraries (benchmarks PUBLIC benchmark::benchmark)

    install (TARGETS benchmarks DESTINATION bin)
endif ()

# --- python_module

if (${BUILD_PYTHON_MODULE})
    add_subdirectory(external_libraries/pybind11)

    add_definitions(
        -DEIGEN_DEFAULT_TO_ROW_MAJOR
        -DHYPERJET_VERSION="${HYPERJET_VERSION}"
        -DHYPERJET_EXCEPTIONS
    )

    pybind11_add_module(python_module src/python_module.cpp)

    set_target_properties(python_module PROPERTIES OUTPUT_NAME hyperjet)

    install(TARGETS python_module DESTINATION bin)
endif ()
