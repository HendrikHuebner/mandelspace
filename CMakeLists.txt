cmake_minimum_required(VERSION 3.12)
project(mandelspace VERSION 1.0 LANGUAGES CXX CUDA)

# Find packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set CUDA compilation flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_70")

# Specify the CUDA source files and their compilation
set(CUDA_SOURCES
    src/core/raymarching.cu
)

# Specify the C++ source files
set(CPP_SOURCES
    src/main.cpp
    src/core/mandelbrot.hpp
    src/util/Config.hpp
)

# Add the executable
add_executable(mandelspace ${CPP_SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(mandelspace 
    ${OPENGL_LIBRARIES} 
    GLEW::GLEW
    glfw
    ${CUDA_LIBRARIES}
)

# Set include directories for the target
target_include_directories(mandelspace PUBLIC 
    ${OPENGL_INCLUDE_DIRS} 
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS} # Use CUDA_INCLUDE_DIRS instead of CUDA_LIBRARIES for include directories
)

# Set properties for CUDA
set_target_properties(mandelspace PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)
