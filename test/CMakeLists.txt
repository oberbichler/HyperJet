cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(HyperJetTest LANGUAGES CXX)

# --- Dependencies

include(../cmake/CPM.cmake)

CPMAddPackage("gh:doctest/doctest@2.4.11")

CPMAddPackage(
  NAME Eigen
  VERSION 3.4.0
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_ONLY YES 
)

if(Eigen_ADDED)
  add_library(Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen SYSTEM INTERFACE ${Eigen_SOURCE_DIR})
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
set_target_properties(HyperJetTest PROPERTIES CXX_STANDARD 23)

# --- Warnings

# HYPERJET_WERROR is enabled by a single CI job. Warnings differ between
# compilers, so promoting them to errors everywhere would let an unrelated
# toolchain block every build.
option(HYPERJET_WERROR "Treat compiler warnings as errors" OFF)

if(MSVC)
  target_compile_options(HyperJetTest PRIVATE /W4)

  if(HYPERJET_WERROR)
    target_compile_options(HyperJetTest PRIVATE /WX)
  endif()
else()
  target_compile_options(HyperJetTest PRIVATE -Wall -Wextra -Wpedantic)

  if(HYPERJET_WERROR)
    target_compile_options(HyperJetTest PRIVATE -Werror)
  endif()
endif()

# --- Sanitizers

option(HYPERJET_SANITIZE "Build the tests with the address and undefined behavior sanitizers" OFF)

if(HYPERJET_SANITIZE)
  target_compile_options(
    HyperJetTest PRIVATE -fsanitize=address,undefined -fno-sanitize-recover=all
                         -fno-omit-frame-pointer -g
  )
  target_link_options(HyperJetTest PRIVATE -fsanitize=address,undefined)
endif()

# --- Add HyperJetTest

enable_testing()

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(HyperJetTest)
