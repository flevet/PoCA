cmake_minimum_required(VERSION 3.25)

# Keep the same source tree usable with the two development environments:
#   legacy : CMake 3.25.x / Visual Studio 2022 / CUDA 11.8
#   modern : CMake >= 3.28 / Visual Studio 2026 / CUDA 13.4
# Do not couple those versions unnecessarily: every compatibility switch below
# is selected from the component that actually requires it.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
    set(POCA_CMAKE_328_PLUS ON)
    cmake_policy(VERSION 3.28)
else()
    set(POCA_CMAKE_328_PLUS OFF)
    cmake_policy(VERSION 3.25)
endif()

project(PoCA LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(POCA_ENABLE_CUDA "Enable CUDA acceleration" ON)
option(POCA_BUILD_EXTRA "Build plugins from the poca_extra repository" ON)

# Validate the optional installed Zarr backend before dependency discovery.
include(cmake/PoCAZarrBackend.cmake)

set(POCA_EXTRA_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../poca_extra/src"
    CACHE PATH
    "Path to the poca_extra source directory")

# Compatibility hint retained from the current main branch. If this directory
# exists it is preferred by FindEigen3; otherwise normal Eigen discovery is used.
if(WIN32 AND EXISTS "C:/Bibliotheques/eigen-5.0.0/Eigen/Core")
    set(_poca_eigen_default "C:/Bibliotheques/eigen-5.0.0")
else()
    set(_poca_eigen_default "")
endif()
set(POCA_EIGEN5_DIR "${_poca_eigen_default}" CACHE PATH
    "Optional Eigen source/install root (legacy name retained for compatibility)")
unset(_poca_eigen_default)

# TinySpline changed the signature used by PoCA between the dependency sets on
# the two machines. Auto-select from the MSVC generation, but keep this cache
# option overridable in case TinySpline is upgraded independently.
set(_poca_tinyspline_new_default OFF)
if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
    set(_poca_tinyspline_new_default ON)
endif()
option(POCA_TINYSPLINE_NEW_API
    "Use the newer TinySpline equidistantKnotSeq(numPoints, numSamples) API"
    ${_poca_tinyspline_new_default})
unset(_poca_tinyspline_new_default)

# Keep build products out of the source checkout. All paths remain cache
# variables, so an existing local build layout can override them if desired.
set(POCA_RUNTIME_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "PoCA executable output directory")
set(POCA_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib" CACHE PATH "PoCA static library output directory")
set(POCA_PLUGIN_OUTPUT_DIR  "${CMAKE_BINARY_DIR}/bin/plugins" CACHE PATH "PoCA plugin output directory")

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

set(POCA_MSVC_2026_PLUS OFF)
if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
    set(POCA_MSVC_2026_PLUS ON)
endif()

# CUDA is optional at configure time. All CUDA-aware targets compile with
# NO_CUDA when it is absent.
set(POCA_HAS_CUDA OFF)
set(POCA_CUDA_13_PLUS OFF)
if(POCA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # CMP0104 (NEW since CMake 3.18) requires every target compiling CUDA
        # sources to have a non-empty CUDA_ARCHITECTURES property.  The legacy
        # VS2022/CUDA 11.8 build can leave CMAKE_CUDA_ARCHITECTURES present but
        # empty in an existing cache, which makes generation fail for every CUDA
        # target.  CMake >= 3.24 supports the special "native" value, so use
        # the GPU(s) of the build machine only when no architecture was supplied
        # explicitly.  Users can still override CMAKE_CUDA_ARCHITECTURES in the
        # cache (for example "75;86" for portable binaries).
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING
                "CUDA architectures to build for (e.g. native or 75;86)" FORCE)
        endif()

        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(POCA_HAS_CUDA ON)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
            set(POCA_CUDA_13_PLUS ON)
        endif()
        message(STATUS "PoCA CUDA support: ON (${CUDAToolkit_VERSION})")
        message(STATUS "PoCA CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(WARNING "POCA_ENABLE_CUDA is ON, but no CUDA compiler was found. Building without CUDA support.")
    endif()
else()
    message(STATUS "PoCA CUDA support: OFF")
endif()

if(POCA_CMAKE_328_PLUS)
    message(STATUS "PoCA CMake profile: >= 3.28 (${CMAKE_VERSION})")
else()
    message(STATUS "PoCA CMake profile: 3.25-compatible (${CMAKE_VERSION})")
endif()
if(MSVC)
    if(POCA_MSVC_2026_PLUS)
        message(STATUS "PoCA MSVC profile: VS 2026-compatible (MSVC_VERSION=${MSVC_VERSION})")
    else()
        message(STATUS "PoCA MSVC profile: VS 2022-compatible (MSVC_VERSION=${MSVC_VERSION})")
    endif()
endif()
if(POCA_HAS_CUDA)
    if(POCA_CUDA_13_PLUS)
        message(STATUS "PoCA CUDA profile: CUDA 13+ compatibility enabled")
    else()
        message(STATUS "PoCA CUDA profile: pre-CUDA-13 compatibility")
    endif()
endif()
message(STATUS "PoCA TinySpline API: ${POCA_TINYSPLINE_NEW_API} (override with -DPOCA_TINYSPLINE_NEW_API=ON/OFF)")

include(cmake/PoCADependencies.cmake)
include(cmake/PoCATargets.cmake)

# Core libraries. Keep this order explicit because it documents the intended architecture.
add_subdirectory(poca_core)
add_subdirectory(poca_geometry)
add_subdirectory(poca_opengl)
add_subdirectory(poca_plot)
add_subdirectory(poca_qt)

# Built-in loaders/plugins.
add_subdirectory(poca_loaderCSVFile)
add_subdirectory(poca_loaderPDBFile)
add_subdirectory(poca_detectionsetplugin)
add_subdirectory(poca_delaunaytriangulationplugin)
add_subdirectory(poca_voronoidiagramplugin)
add_subdirectory(poca_objectlistplugin)
add_subdirectory(poca_coloctesselerplugin)
add_subdirectory(poca_kripleyplugin)
add_subdirectory(poca_dbscanplugin)
add_subdirectory(poca_nearestlocsmulticolorplugin)
add_subdirectory(poca_objectcolocalizationplugin)

# Main executable after all public PoCA library targets exist.
add_subdirectory(poca)

if(POCA_BUILD_EXTRA)
    if(EXISTS "${POCA_EXTRA_SOURCE_DIR}/CMakeLists.txt")
        add_subdirectory("${POCA_EXTRA_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/poca_extra")
    else()
        message(WARNING
            "POCA_BUILD_EXTRA is ON but poca_extra was not found at: ${POCA_EXTRA_SOURCE_DIR}. "
            "Set POCA_EXTRA_SOURCE_DIR or disable POCA_BUILD_EXTRA.")
    endif()
endif()

if(MSVC)
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT poca)
endif()
