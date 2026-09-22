cmake_minimum_required(VERSION 3.6)
project(poca)

if(POLICY CMP0020)
	cmake_policy(SET CMP0020 OLD)
endif()

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


#set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../bin/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../bin/plugins")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../bin")

# Eigen is kept as a single project-wide dependency so PoCA, CGAL-facing code
# and optional plugins all compile against the same headers. The directory is
# the extracted Eigen source tree; no Eigen build/install step is required.
if(WIN32)
    set(_POCA_EIGEN5_DEFAULT "C:/Bibliotheques/eigen-5.0.0")
else()
    set(_POCA_EIGEN5_DEFAULT "")
endif()
set(POCA_EIGEN5_DIR "${_POCA_EIGEN5_DEFAULT}" CACHE PATH "Path to the Eigen 5.0.0 source directory")
unset(_POCA_EIGEN5_DEFAULT)

if(NOT EXISTS "${POCA_EIGEN5_DIR}/Eigen/Core" OR NOT EXISTS "${POCA_EIGEN5_DIR}/Eigen/Version")
    message(FATAL_ERROR "Eigen 5.0.0 was not found in '${POCA_EIGEN5_DIR}'. Set POCA_EIGEN5_DIR to the extracted Eigen 5.0.0 directory.")
endif()
file(STRINGS "${POCA_EIGEN5_DIR}/Eigen/Version" _POCA_EIGEN_VERSION_LINE REGEX "^#define EIGEN_VERSION_STRING")
if(NOT _POCA_EIGEN_VERSION_LINE STREQUAL "#define EIGEN_VERSION_STRING \"5.0.0\"")
    message(FATAL_ERROR "POCA_EIGEN5_DIR must point to Eigen 5.0.0. Found version declaration: ${_POCA_EIGEN_VERSION_LINE}")
endif()
unset(_POCA_EIGEN_VERSION_LINE)

message(STATUS "Using Eigen 5.0.0 from ${POCA_EIGEN5_DIR}")
include_directories(SYSTEM "${POCA_EIGEN5_DIR}")

# Provide the modern imported target as well, without requiring Eigen to be
# installed as a CMake package. This also gives downstream PoCA targets one
# unambiguous Eigen include directory.
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${POCA_EIGEN5_DIR}")
endif()
set(Eigen3_FOUND TRUE)
set(EIGEN3_FOUND TRUE)
set(Eigen3_VERSION "5.0.0")
set(EIGEN3_VERSION_STRING "5.0.0")
set(EIGEN3_INCLUDE_DIR "${POCA_EIGEN5_DIR}")
set(EIGEN3_INCLUDE_DIRS "${POCA_EIGEN5_DIR}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
"${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(ADD_PYTHON True)

add_subdirectory(poca_core)
add_subdirectory(poca_opengl)
add_subdirectory(poca_geometry)
add_subdirectory(poca_plot)
add_subdirectory(poca_qt)
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
add_subdirectory(poca)

add_dependencies(poca_geometry poca_core)
add_dependencies(poca_qt poca_core poca_plot)
add_dependencies(poca_opengl poca_core poca_geometry)
add_dependencies(poca_loadercsv poca_core poca_geometry)
add_dependencies(poca_loaderpdb poca_core poca_geometry)
add_dependencies(poca_detectionsetplugin poca_core poca_opengl poca_plot poca_geometry)
add_dependencies(poca_delaunaytriangulationplugin poca_core poca_opengl poca_plot poca_geometry)
add_dependencies(poca_voronoidiagramplugin poca_core poca_opengl poca_plot poca_geometry)
add_dependencies(poca_objectlistplugin poca_core poca_opengl poca_plot poca_geometry)
add_dependencies(poca_coloctesselerplugin poca_core poca_opengl poca_plot poca_geometry)
add_dependencies(poca_kripleyplugin poca_core poca_opengl poca_plot poca_geometry)
add_dependencies(poca_dbscanplugin poca_core poca_opengl poca_plot poca_geometry)
add_dependencies(poca_nearestlocsmulticolorplugin poca_core poca_opengl poca_plot poca_geometry)
add_dependencies(poca_objectcolocalizationplugin poca_core poca_opengl poca_plot poca_geometry)
add_dependencies(poca poca_core poca_qt poca_opengl poca_plot poca_geometry)

add_subdirectory(../../poca_extra/src ${CMAKE_BINARY_DIR}/extra)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT poca)
message("Last one ? C++ flags: ${CMAKE_CXX_FLAGS}")
