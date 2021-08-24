cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(Benchmark LANGUAGES CXX)

# --- Dependencies

include(../cmake/CPM.cmake)

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

set_target_properties(Benchmark PROPERTIES CXX_STANDARD 17)