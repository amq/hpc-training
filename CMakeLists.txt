cmake_minimum_required(VERSION 3.9.0)
project(GaussianBlur VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()

add_executable(GaussianBlur main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package(OpenCL 1.2 REQUIRED)
target_link_libraries(GaussianBlur OpenCL::OpenCL)

target_include_directories(GaussianBlur PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/OpenCL-CLHPP/include)
