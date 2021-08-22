cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(BuildAll LANGUAGES CXX)

enable_testing()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../benchmark ${CMAKE_BINARY_DIR}/benchmark)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../test ${CMAKE_BINARY_DIR}/test)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../python ${CMAKE_BINARY_DIR}/python)