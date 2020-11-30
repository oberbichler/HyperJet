cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

project(hyperjet)

include_directories(
    "include"
    "external_libraries"
)

# --- tests

add_executable(tests src/tests.cpp)

install(TARGETS tests DESTINATION bin)

# --- benchmarks

add_executable(benchmarks src/benchmarks.cpp)

install(TARGETS benchmarks DESTINATION bin)

# --- python_module

add_subdirectory(external_libraries/pybind11)

add_definitions(
    -DEIGEN_DEFAULT_TO_ROW_MAJOR
    -DHYPERJET_VERSION="${HYPERJET_VERSION}"
    -DHYPERJET_EXCEPTIONS
)

pybind11_add_module(python_module src/python_module.cpp)

set_target_properties(python_module PROPERTIES OUTPUT_NAME hyperjet)

install(TARGETS python_module DESTINATION bin)