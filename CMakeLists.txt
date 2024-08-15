cmake_minimum_required(VERSION 3.12)
project(mandelspace VERSION 1.0)

# find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/util/Config.hpp
)

add_executable(mandelspace ${SOURCES})

# target_link_libraries(mandelspace ${CUDA_LIBRARIES})
