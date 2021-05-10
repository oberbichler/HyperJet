cmake_minimum_required(VERSION 3.12)

if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "")
endif()

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

message(NOTICE "VCPKG_TARGET_TRIPLET: ${VCPKG_TARGET_TRIPLET}")
message(NOTICE "VCPKG_ROOT: ${VCPKG_ROOT}")
message(NOTICE "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")

project(hyperjet)

option(BUILD_PYTHON_MODULE "Build the Python module" ON )
option(BUILD_TESTS         "Build the C++ tests"     ON )
option(BUILD_BENCHMARKS    "Build the benchmarks"    OFF)

include_directories("include" "external_libraries")

# --- tests

if (${BUILD_TESTS})
    find_package(Catch2 CONFIG REQUIRED)
    find_package(Eigen3 REQUIRED)

    add_executable(tests src/tests.cpp)

    target_compile_features(tests PRIVATE cxx_std_17)

    target_link_libraries(tests PRIVATE Catch2::Catch2)
    target_link_libraries(tests PRIVATE Eigen3::Eigen)

    install(TARGETS tests DESTINATION bin)
endif()

# --- benchmarks

if (${BUILD_BENCHMARKS})
    find_package(benchmark REQUIRED)
    find_package(Eigen3 REQUIRED)

    add_executable(benchmarks src/benchmarks.cpp)

    target_compile_features(benchmarks PRIVATE cxx_std_17)

    target_link_libraries(benchmarks PRIVATE benchmark::benchmark)
    target_link_libraries(benchmarks PRIVATE Eigen3::Eigen)

    install(TARGETS benchmarks DESTINATION bin)
endif()

# --- python_module

if (${BUILD_PYTHON_MODULE})
    add_subdirectory(external_libraries/pybind11)

    find_package(Eigen3 REQUIRED)

    pybind11_add_module(python_module src/python_module.cpp)

    target_compile_definitions(python_module PRIVATE
        -DEIGEN_DEFAULT_TO_ROW_MAJOR
        -DHYPERJET_VERSION="${HYPERJET_VERSION}"
        -DHYPERJET_EXCEPTIONS
    )

    target_compile_features(python_module PRIVATE cxx_std_17)

    set_target_properties(python_module PROPERTIES OUTPUT_NAME hyperjet)

    target_link_libraries(python_module PRIVATE Eigen3::Eigen)

    install(TARGETS python_module DESTINATION bin)
endif()
