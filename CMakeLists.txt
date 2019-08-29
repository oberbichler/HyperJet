cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

project(HyperJet-py)

find_package(BLAS)

include_directories(
    "include"
    "external_libraries"
)

add_subdirectory(external_libraries/pybind11)

add_definitions(
    -DEIGEN_DEFAULT_TO_ROW_MAJOR
    -DHYPERJET_VERSION="${HYPERJET_VERSION}"
    -DHYPERJET_EXCEPTIONS
)

pybind11_add_module(hyperjet src/Module.cc)

if(BLAS_FOUND)
    add_definitions(-DEIGEN_USE_BLAS)
    target_link_libraries(hyperjet PRIVATE ${BLAS_LIBRARIES})
endif()

install(TARGETS hyperjet DESTINATION bin)