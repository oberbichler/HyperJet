cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)

project(HyperJet-py)

include_directories(
    "include"
    "external_libraries"
)

add_subdirectory(external_libraries/pybind11)

add_definitions(
    -DEIGEN_DEFAULT_TO_ROW_MAJOR
    -DHYPERJET_VERSION="${HYPERJET_VERSION}"
)

pybind11_add_module(HyperJet src/Module.cc)

install(TARGETS HyperJet DESTINATION bin)