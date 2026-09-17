cmake_minimum_required(VERSION 3.28)

project(PoCA LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(POCA_ENABLE_CUDA "Enable CUDA acceleration" ON)
option(POCA_BUILD_EXTRA "Build plugins from the poca_extra repository" ON)

set(POCA_EXTRA_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../poca_extra/src"
    CACHE PATH
    "Path to the poca_extra source directory")

# Keep build products out of the source checkout.
set(POCA_RUNTIME_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "PoCA executable output directory")
set(POCA_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib" CACHE PATH "PoCA static library output directory")
set(POCA_PLUGIN_OUTPUT_DIR  "${CMAKE_BINARY_DIR}/bin/plugins" CACHE PATH "PoCA plugin output directory")

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# CUDA is optional at configure time.  All targets compile with NO_CUDA when it is absent.
set(POCA_HAS_CUDA OFF)
if(POCA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(POCA_HAS_CUDA ON)
        message(STATUS "PoCA CUDA support: ON (${CUDAToolkit_VERSION})")
    else()
        message(WARNING "POCA_ENABLE_CUDA is ON, but no CUDA compiler was found. Building without CUDA support.")
    endif()
else()
    message(STATUS "PoCA CUDA support: OFF")
endif()

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
