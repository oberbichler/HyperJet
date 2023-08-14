cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(HyperJetTest LANGUAGES CXX)

# --- Dependencies

include(../cmake/CPM.cmake)

CPMAddPackage("gh:onqtam/doctest#2.4.11")

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

if(TEST_INSTALLED_VERSION)
  find_package(HyperJet REQUIRED)
else()
  CPMAddPackage(NAME HyperJet SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()

# --- Create binary

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(HyperJetTest ${sources})
target_link_libraries(HyperJetTest doctest::doctest hyperjet::hyperjet Eigen)
set_target_properties(HyperJetTest PROPERTIES CXX_STANDARD 17)

# --- Add HyperJetTest

enable_testing()

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(HyperJetTest)
