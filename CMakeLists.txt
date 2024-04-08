cmake_minimum_required(VERSION 3.24)
project(CUDA_Median_Filter CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

set(CMAKE_TOOLCHAIN_FILE "C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake")
include(${CMAKE_TOOLCHAIN_FILE})

find_package(OpenCV CONFIG REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cu)

set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(main PRIVATE ${OpenCV_LIBS})
