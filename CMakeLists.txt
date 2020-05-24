cmake_minimum_required(VERSION 3.11.0)
project(GaussianBlur
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

include(FetchContent)

FetchContent_Declare(
  opencl-clhpp
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
  GIT_TAG        v2.0.11
)

FetchContent_GetProperties(opencl-clhpp)
if(NOT opencl-clhpp_POPULATED)
  FetchContent_Populate(opencl-clhpp)
endif()

FetchContent_Declare(
  opencl-headers
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
  GIT_TAG        v2020.03.13
)

FetchContent_GetProperties(opencl-headers)
if(NOT opencl-headers_POPULATED)
  FetchContent_Populate(opencl-headers)
endif()

FetchContent_Declare(
  opencl-icd-loader
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
  GIT_TAG        v2020.03.13
)

FetchContent_GetProperties(opencl-icd-loader)
if(NOT opencl-icd-loader_POPULATED)
  FetchContent_Populate(opencl-icd-loader)
  set(OPENCL_ICD_LOADER_REQUIRE_WDK OFF CACHE BOOL "Build with D3DKMT support, which requires the Windows WDK")
  set(OPENCL_ICD_LOADER_HEADERS_DIR ${opencl-headers_SOURCE_DIR} CACHE STRING "Path to OpenCL Headers")
  add_subdirectory(${opencl-icd-loader_SOURCE_DIR} ${opencl-icd-loader_BINARY_DIR})
endif()

FetchContent_Declare(
  lodepng
  GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
  GIT_TAG        486d165ed70999cd575a9996a7f2551c7b238c81
)

FetchContent_GetProperties(lodepng)
if(NOT lodepng_POPULATED)
  FetchContent_Populate(lodepng)
endif()

add_library(lodepng STATIC ${lodepng_SOURCE_DIR}/lodepng.cpp ${lodepng_SOURCE_DIR}/lodepng.h)

add_executable(GaussianBlur main.cpp filters.hpp)
target_link_libraries(GaussianBlur PRIVATE OpenCL lodepng)
target_include_directories(GaussianBlur PRIVATE
                           ${opencl-clhpp_SOURCE_DIR}/include
                           ${opencl-headers_SOURCE_DIR}
                           ${lodepng_SOURCE_DIR})
