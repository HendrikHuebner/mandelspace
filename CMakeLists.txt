cmake_minimum_required(VERSION 3.12)
project(mandelspace VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
# find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SOURCES
src/main.cpp
src/core/mandelbrot.hpp
src/util/Config.hpp
)

add_executable(mandelspace ${SOURCES})

target_link_libraries(mandelspace 
    ${OPENGL_LIBRARIES} 
    GLEW::GLEW 
    glfw
    # ${CUDA_LIBRARIES}
)

target_include_directories(mandelspace PUBLIC 
    ${OPENGL_INCLUDE_DIRS} 
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
)


