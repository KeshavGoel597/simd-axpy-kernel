cmake_minimum_required(VERSION 3.11...3.23)
set(CMAKE_BUILD_TYPE "Release")

project(
    AXPY_ASSIGNMENT
    VERSION 1
    DESCRIPTION "An assignment on optimizing AXPY"
    LANGUAGES CXX)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(src)
add_subdirectory(tester)
