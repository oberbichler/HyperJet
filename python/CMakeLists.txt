cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(HyperJetPython LANGUAGES CXX)

# --- Dependencies

include(../cmake/CPM.cmake)

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

CPMAddPackage("gh:/pybind/pybind11@2.11.1")

if(TEST_INSTALLED_VERSION)
  find_package(HyperJet REQUIRED)
else()
  CPMAddPackage(NAME HyperJet SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()

# --- Create binary

include_directories("external_libraries")

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
pybind11_add_module(HyperJetPython ${sources})

target_link_libraries(HyperJetPython PRIVATE hyperjet::hyperjet Eigen)

target_compile_definitions(HyperJetPython PRIVATE
  -DHYPERJET_VERSION="${PROJECT_VERSION}"
  -DHYPERJET_EXCEPTIONS
)

set_target_properties(HyperJetPython PROPERTIES CXX_STANDARD 17)

set_target_properties(HyperJetPython PROPERTIES OUTPUT_NAME hyperjet)
