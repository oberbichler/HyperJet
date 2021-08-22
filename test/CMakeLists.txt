cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(HyperJetTest LANGUAGES CXX)

# ---- Dependencies

include(../cmake/CPM.cmake)

CPMAddPackage("gh:onqtam/doctest#2.4.5")

if(TEST_INSTALLED_VERSION)
  find_package(HyperJet REQUIRED)
else()
  CPMAddPackage(NAME HyperJet SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()

# ---- Create binary ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(HyperJetTest ${sources})
target_link_libraries(HyperJetTest doctest::doctest hyperjet::hyperjet)
set_target_properties(HyperJetTest PROPERTIES CXX_STANDARD 17)

# # enable compiler warnings
# if(NOT TEST_INSTALLED_VERSION)
#   if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
#     target_compile_options(HyperJet PUBLIC -Wall -Wpedantic -Wextra -Werror)
#   elseif(MSVC)
#     target_compile_options(HyperJet PUBLIC /W4 /WX)
#     target_compile_definitions(HyperJetTest PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
#   endif()
# endif()

# ---- Add HyperJetTest ----

enable_testing()

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(HyperJetTest)

# ---- code coverage ----

if(ENABLE_TEST_COVERAGE)
  target_compile_options(HyperJet PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(HyperJet PUBLIC -fprofile-arcs -ftest-coverage)
endif()