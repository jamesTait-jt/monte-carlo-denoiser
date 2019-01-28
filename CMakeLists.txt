cmake_minimum_required(VERSION 3.13.1)
project (mcr)

# Set threads flag on
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Set output directory for executables
set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_HOME_DIRECTORY}/build
)

# Set the output directory for library:
set(
    CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_HOME_DIRECTORY}/lib
)

# Find the required libraries
find_package(SDL2 REQUIRED)
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

# Bring the headers into the project
include_directories(
    include 
    ${SDL2_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)

# Set the sources
set(
    SOURCES 
    src/mcr.cc
    src/sdl_window_helper.cc 
    src/shape.cc
    src/triangle.cc
    src/material.cc
    src/ray.cc
)

add_executable(${PROJECT_NAME} 
    ${SOURCES}
)

set(
    SOURCES
    test/tests.cc
    src/sdl_window_helper.cc 
    src/shape.cc
    src/triangle.cc
    src/material.cc
    src/ray.cc
)

add_executable(run_tests
    ${SOURCES}
)

# Link threads
target_link_libraries(${PROJECT_NAME} SDL2::SDL2 Threads::Threads)
target_link_libraries(run_tests ${GTEST_LIBRARIES} SDL2::SDL2 Threads::Threads)
