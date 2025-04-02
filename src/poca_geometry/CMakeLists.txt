#if(WIN32)
#  set(CMAKE_WIN32_EXECUTABLE ON)
#endif()

# This code sorts the project files as they appear in the root directory

# Generate a list of all .c & .h files in the current directory and sub directores.

set(targetname "poca_geometry")
message("***************** Processing ${targetname}")

project(${targetname})

include_directories(${CMAKE_SOURCE_DIR}/poca_core)
link_directories(${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

set(3D_VORONOI_GPU_HEADER_FILES
  3D_voronoi_GPU/voronoi.h
  )

set(INTERFACES_HEADER_FILES
  Interfaces/ObjectFeaturesFactoryInterface.hpp
  Interfaces/ObjectListFactoryInterface.hpp
  Interfaces/ObjectListInterface.hpp
  Interfaces/ObjectIndicesFactoryInterface.hpp
  Interfaces/VoronoiDiagramFactoryInterface.hpp
  Interfaces/DelaunayTriangulationFactoryInterface.hpp
  Interfaces/DelaunayTriangulationInterface.hpp
  Interfaces/PolygonFactoryInterface.hpp
  Interfaces/PolygonInterface.hpp
)

set(FACTORY_HEADER_FILES
  Factory/ObjectListFactory.hpp
  Factory/ObjectFeaturesFactory.hpp
  Factory/VoronoiDiagramFactory.hpp
  Factory/DelaunayTriangulationFactory.hpp
  Factory/PolygonFactory.hpp
  Factory/ObjectIndicesFactory.hpp
)

set(FACTORY_SOURCE_FILES
  Factory/ObjectListFactory.cpp
  Factory/ObjectFeaturesFactory.cpp
  Factory/VoronoiDiagramFactory.cpp
  Factory/DelaunayTriangulationFactory.cpp
  Factory/PolygonFactory.cpp
  Factory/ObjectIndicesFactory.cpp
)

set(GEOMETRY_HEADER_FILES
  Geometry/BasicComputation.hpp
  Geometry/DelaunayTriangulation.hpp
  Geometry/DetectionSet.hpp
  Geometry/nanoflann.hpp
  Geometry/ObjectLists.hpp
  Geometry/ObjectListDelaunay.hpp
  Geometry/ObjectListMesh.hpp
  Geometry/StraightLine.hpp
  Geometry/Skeleton.hpp
  Geometry/VoronoiDiagram.hpp
  Geometry/CGAL_includes.hpp
  Geometry/delaunator.hpp
  Geometry/voronator.hpp
  Geometry/dbscan.h
  Geometry/kdtree.h
  Geometry/Polygon.hpp
  Geometry/DelaunayOnSphere.hpp
  Geometry/TrackSet.hpp
  )

set(GEOMETRY_SOURCE_FILES
  Geometry/BasicComputation.cpp
  Geometry/DelaunayTriangulation.cpp
  Geometry/DetectionSet.cpp
  Geometry/ObjectLists.cpp
  Geometry/ObjectListDelaunay.cpp
  Geometry/ObjectListMesh.cpp
  Geometry/StraightLine.cpp  
  Geometry/Skeleton.cpp
  Geometry/VoronoiDiagram.cpp
  Geometry/delaunator.cpp  
  Geometry/voronator.cpp
  Geometry/kdtree.cpp
  Geometry/Polygon.cpp
  Geometry/DelaunayOnSphere.cpp
  Geometry/TrackSet.cpp
)

set(header_list
  ${3D_VORONOI_GPU_HEADER_FILES}
  ${INTERFACES_HEADER_FILES}
  ${FACTORY_HEADER_FILES}
  ${GEOMETRY_HEADER_FILES}
  )

set(source_list
  ${FACTORY_SOURCE_FILES}
  ${GEOMETRY_SOURCE_FILES}
  )
  
 set(cuda_list
  3D_voronoi_GPU/voronoi.cu
  )
  
foreach(source IN LISTS header_list)
    get_filename_component(source_path "${source}" PATH)
    #message(STATUS ${source_path})
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    set(source_path_msvc "${source_path_msvc}\\Header Files")
    #message(STATUS ${source_path_msvc})
    source_group("${source_path_msvc}" FILES "${source}")
  endforeach()
foreach(source IN LISTS source_list)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    set(source_path_msvc "${source_path_msvc}\\Source Files")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()

find_package(CUDA QUIET)

find_package(Eigen3 3.1.0)
  if (EIGEN3_FOUND)
  include( ${EIGEN3_USE_FILE} )
endif()

find_package(CGAL REQUIRED COMPONENTS Core)
if(CGAL_FOUND)
    message("Found CGAL.")
    if(CGAL_VERSION VERSION_LESS "6.0")
	message("CGAL version 5")
	include(${CGAL_USE_FILE})
    	message("CGAL thing: ${CGAL_USE_FILE}")

    	message("CGAL_CXX_FLAGS_INIT: ${CGAL_CXX_FLAGS_INIT}")
    	add_definitions(${CGAL_CXX_GLAFS_INIT})

    	message("CGAL_LIBRARIES: ${CGAL_LIBRARY_DIRS}")
    	message("CGAL_3RD_PARTY_LIBRARIES: ${CGAL_3RD_PARTY_LIBRARIES_DIRS}")
    endif()
else()
    message(FATAL_ERROR "CGAL is required.")
endif()

add_definitions(-DNOMINMAX)
add_definitions(-DQT_NO_WARNING_OUTPUT)
find_package(Qt5 COMPONENTS Core REQUIRED)

if(CUDA_FOUND)
  message("CUDA was found")
  #set(CMAKE_CUDA_FLAGS "-Xcompiler=\"/EHsc -Zi -Ob0\"")
  message("CUDA flags: ${CMAKE_CUDA_FLAGS}")
  enable_language(CUDA)
	#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
	foreach(source IN LISTS cuda_list)
		get_filename_component(source_path "${source}" PATH)
		string(REPLACE "/" "\\" source_path_msvc "${source_path}")
		message(STATUS ${source_path_msvc})
		source_group("${source_path_msvc}" FILES "${source}")
	endforeach()
	add_library(${targetname} STATIC ${header_list} ${source_list} ${cuda_list})
	#target_compile_options(poca PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:"-Xcompiler="/EHsc -Zi -Ob0 -Ob2"">) 
else()
	message("CUDA not found")
	add_definitions(-DNO_CUDA)
	add_library(${targetname} STATIC ${header_list} ${source_list})
endif()

if(CGAL_FOUND)
    message("Found CGAL.")
    if(CGAL_VERSION VERSION_GREATER_EQUAL "6.0")
	message("CGAL version 6")
	target_link_libraries(${targetname} CGAL::CGAL)
    endif()
endif()

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})
target_link_libraries(${targetname} ${GLEW_STATIC_LIBRARIES})

find_package(TBB COMPONENTS tbb tbbmalloc)
if(TBB_FOUND)
  include_directories(PRIVATE ${TBB_INCLUDE_DIRS})
  target_compile_definitions(${targetname} PRIVATE ${TBB_DEFINITIONS})
  target_link_libraries(${targetname} ${TBB_IMPORTED_TARGETS})
  target_link_libraries(${targetname} debug ${TBB_tbb_LIBRARY_DEBUG} debug ${TBB_tbbmalloc_LIBRARY_DEBUG})
  target_link_libraries(${targetname} optimized ${TBB_tbb_LIBRARY_RELEASE} optimized ${TBB_tbbmalloc_LIBRARY_RELEASE})
  add_definitions(-DCGAL_LINKED_WITH_TBB)
  message("TBB: ${TBB_tbb_LIBRARY_DEBUG}")
  message("TBB: ${TBB_tbbmalloc_LIBRARY_DEBUG}")
  message("TBB: ${TBB_tbb_LIBRARY_RELEASE}")
  message("TBB: ${TBB_tbbmalloc_LIBRARY_RELEASE}")
  message("TBB: ${TBB_INCLUDE_DIRS}")
  message("TBB: ${TBB_IMPORTED_TARGETS}")
  #message("TBB2: ${TBB_tbb_LIBRARY_DEBUG}")
endif()

target_link_libraries(${targetname} Qt5::Core)

#add_library(${targetname} STATIC ${header_list} ${source_list})
#set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${targetname})
set_target_properties(${targetname} PROPERTIES OUTPUT_NAME "${targetname}$<$<CONFIG:Debug>:d>")
set_target_properties(${targetname} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(youroutputdirectory "${CMAKE_SOURCE_DIR}/../bin/lib")
set_target_properties( ${targetname} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${youroutputdirectory} )
set_target_properties( ${targetname} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${youroutputdirectory} )
set_target_properties( ${targetname} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${youroutputdirectory} )

set_property(TARGET ${targetname} PROPERTY CXX_STANDARD 17)
