cmake_minimum_required(VERSION 3.10)
project(libvirtualscanner VERSION 18.10.09 LANGUAGES CXX)

if(NOT VCPKG_TOOLCHAIN)
    list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()

##################################################
# Dependencies
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)

# Try to find CGAL using the config mode first
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{CONDA_PREFIX}/lib/cmake/CGAL)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(virtualscanner PUBLIC cxx_nullptr)
target_compile_options(virtualscanner
    PRIVATE 
        $<$<CXX_COMPILER_ID:GNU>:-Wall -fPIC -Wno-sign-compare -Wno-uninitialized -frounding-math>
        $<$<CXX_COMPILER_ID:MSVC>: /Wall>)

target_link_libraries(virtualscanner
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        Boost::system
        CGAL::CGAL)
