cmake_minimum_required(VERSION 3.9.0)
project(GaussianBlur VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()

add_executable(GaussianBlur src/main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(OPENCL_ICD_LOADER_REQUIRE_WDK OFF CACHE BOOL "Build with D3DKMT support, which requires the Windows WDK")
set(OPENCL_ICD_LOADER_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenCL-Headers CACHE STRING "Path to OpenCL Headers")
add_subdirectory(external/OpenCL-ICD-Loader)

target_link_libraries(GaussianBlur OpenCL)
target_include_directories(GaussianBlur PRIVATE
                          ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenCL-Headers
                          ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenCL-CLHPP/include
                          ${CMAKE_CURRENT_SOURCE_DIR}/external/CImg)
