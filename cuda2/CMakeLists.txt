cmake_minimum_required(VERSION 3.13.1)
project (cuda-mcr LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-pipe -O3 -Wall -Wno-switch -g")
#message(${CMAKE_CXX_FLAGS})

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

include_directories(
    include
    ${SDL2_INCLUDE_DIRS}
)

set(
    SOURCES
    src/mcr.cu
    src/sdl_window_helper.cc
    src/shape.cc
    src/triangle.cu
    src/material.cc
    src/ray.cu
    src/camera.cc
    src/light.cu
    src/lightSphere.cu
)

add_executable(cuda_mcr 
    ${SOURCES}
)

# Request that particles be built with --std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(cuda_mcr PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(cuda_mcr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda_mcr PROPERTIES LINKER_LANGUAGE CXX)

# Link libraries
target_link_libraries(cuda_mcr SDL2::SDL2)
