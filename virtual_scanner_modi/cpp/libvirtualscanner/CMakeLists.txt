cmake_minimum_required(VERSION 3.10)
project(libvirtualscanner VERSION 18.10.09 LANGUAGES CXX)

if(NOT VCPKG_TOOLCHAIN)
    list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()

##################################################
# Dependencies
find_package(Boost REQUIRED COMPONENTS system)

# 明确设置Eigen3的路径
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
if(EXISTS "${EIGEN3_INCLUDE_DIR}")
    message(STATUS "找到Eigen3库路径: ${EIGEN3_INCLUDE_DIR}")
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    message(FATAL_ERROR "找不到Eigen3库，请安装: sudo apt-get install libeigen3-dev")
endif()

# 设置CGAL路径
set(CGAL_DIR "/usr/lib/x86_64-linux-gnu/cmake/CGAL")
if(EXISTS "${CGAL_DIR}")
    message(STATUS "找到CGAL路径: ${CGAL_DIR}")
else()
    message(FATAL_ERROR "找不到CGAL库，请安装: sudo apt-get install libcgal-dev")
endif()

# Try to find CGAL using the config mode first
find_package(CGAL CONFIG QUIET)
if(NOT CGAL_FOUND)
    # Fall back to module mode
    find_package(CGAL REQUIRED)
endif()
##################################################

add_library(virtualscanner
    src/virtual_scanner.cpp
    src/points.cpp)

add_library(OCNN::VirtualScanner ALIAS virtualscanner)

target_include_directories(virtualscanner
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${EIGEN3_INCLUDE_DIR})

target_compile_features(virtualscanner PUBLIC cxx_nullptr)
target_compile_options(virtualscanner
    PRIVATE 
        $<$<CXX_COMPILER_ID:GNU>:-Wall -fPIC -Wno-sign-compare -Wno-uninitialized -frounding-math>
        $<$<CXX_COMPILER_ID:MSVC>: /Wall>)

target_link_libraries(virtualscanner
    PRIVATE
        Boost::system
        CGAL::CGAL)
