cmake_minimum_required(VERSION 3.12)
project(mandelspace VERSION 1.0 LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(glm REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_70 -gencode arch=compute_70,code=sm_70 --use_fast_math ")

set(CUDA_SOURCES
    src/cuda/raymarching.cu
    src/cuda/fractals/menger_prison.cu
    src/cuda/fractals/mandelbox.cu
    src/cuda/fractals/mandelbulb.cu
)

set(CPP_SOURCES
    src/main.cpp
    src/window.hpp
    src/ui.hpp
    src/util/Config.hpp
)

add_executable(mandelspace ${CPP_SOURCES} ${CUDA_SOURCES})

target_link_libraries(mandelspace 
    ${OPENGL_LIBRARIES} 
    GLEW::GLEW
    glm::glm
    glfw
    ${CUDA_LIBRARIES}
)

target_include_directories(mandelspace PUBLIC 
    ${OPENGL_INCLUDE_DIRS} 
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

set_target_properties(mandelspace PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)
