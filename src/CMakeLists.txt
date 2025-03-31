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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
"${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(ADD_PYTHON True)

add_subdirectory(poca_core)
add_subdirectory(poca_opengl)
add_subdirectory(poca_geometry)
add_subdirectory(poca_plot)
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
add_dependencies(poca poca_core poca_opengl poca_plot poca_geometry)

add_subdirectory(../../poca_extra/src ${CMAKE_BINARY_DIR}/extra)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT poca)
message("Last one ? C++ flags: ${CMAKE_CXX_FLAGS}")