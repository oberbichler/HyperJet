cmake_minimum_required (VERSION 3.12)

project (hyperjet)

option (BUILD_PYTHON_MODULE "Build the Python module" ON )
option (BUILD_TESTS         "Build the C++ tests"     ON )
option (BUILD_BENCHMARKS    "Build the benchmarks"    OFF)

include_directories ("include")

# --- tests

if (${BUILD_TESTS})
    find_package (catch2 REQUIRED)
    find_package (Eigen3 REQUIRED)

    add_executable (tests src/tests.cpp)

    target_compile_features (tests PRIVATE cxx_std_17)

    target_link_libraries (tests PRIVATE Catch2::Catch2 Eigen3::Eigen)

    install (TARGETS tests DESTINATION bin)
endif ()

# --- benchmarks

if (${BUILD_BENCHMARKS})
    find_package (benchmark REQUIRED)

    add_executable (benchmarks src/benchmarks.cpp)

    target_compile_features (benchmarks PRIVATE cxx_std_17)

    target_link_libraries (benchmarks PRIVATE benchmark::benchmark)

    install (TARGETS benchmarks DESTINATION bin)
endif ()

# --- python_module

if (${BUILD_PYTHON_MODULE})
    find_package (pybind11 REQUIRED)
    find_package (Eigen3 REQUIRED)

    pybind11_add_module (python_module src/python_module.cpp)

    target_compile_definitions (python_module PRIVATE
        -DEIGEN_DEFAULT_TO_ROW_MAJOR
        -DHYPERJET_VERSION="${HYPERJET_VERSION}"
        -DHYPERJET_EXCEPTIONS
    )

    target_compile_features (python_module PRIVATE cxx_std_17)

    set_target_properties (python_module PROPERTIES OUTPUT_NAME hyperjet)

    target_link_libraries (python_module PRIVATE Eigen3::Eigen)

    install (TARGETS python_module DESTINATION bin)
endif ()
