cmake_minimum_required(VERSION 3.22)

project(lab1)
SET(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_TOOLCHAIN_FILE ~/vcpkg/scripts/buildsystems/vcpkg.cmake)

add_executable(lab1 src/main.cpp)
find_package(OpenCL REQUIRED)
target_link_libraries(lab1 PRIVATE OpenCL::OpenCL)

