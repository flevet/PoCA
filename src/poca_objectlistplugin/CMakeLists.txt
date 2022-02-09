#if(WIN32)
#  set(CMAKE_WIN32_EXECUTABLE ON)
#endif()

# This code sorts the project files as they appear in the root directory

# Generate a list of all .c & .h files in the current directory and sub directores.

set(targetname "poca_objectlistplugin")
message("***************** Processing ${targetname}")

project(${targetname})

include_directories(${CMAKE_SOURCE_DIR}/poca_core)
include_directories(${CMAKE_SOURCE_DIR}/poca_opengl)
include_directories(${CMAKE_SOURCE_DIR}/poca_plot)
include_directories(${CMAKE_SOURCE_DIR}/poca_geometry)
include_directories(${CMAKE_SOURCE_DIR}/../include)
link_directories(${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

set(OBJECT_LIST_PLUGIN_HEADER_FILES
  ObjectListBasicCommands.hpp
  ObjectListPlugin.hpp
  ObjectListDisplayCommand.hpp
  ObjectListWidget.hpp
  ObjectListParamDialog.hpp
  )

set(OBJECT_LIST_PLUGIN_SOURCE_HEADER_FILES
  ObjectListBasicCommands.cpp
  ObjectListPlugin.cpp
  ObjectListDisplayCommand.cpp
  ObjectListWidget.cpp
  ObjectListParamDialog.cpp
)

set(FOO_MOC_HEADERS
  ObjectListPlugin.hpp
  ObjectListWidget.hpp
  ObjectListParamDialog.hpp
)

set(FOO_SOURCE_FILES
  ObjectListPlugin.cpp
  ObjectListWidget.cpp
  ObjectListParamDialog.cpp
)

set(header_list
  ${OBJECT_LIST_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${OBJECT_LIST_PLUGIN_SOURCE_HEADER_FILES}
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

add_definitions(-DNOMINMAX)
add_definitions(-DQT_NO_WARNING_OUTPUT)
find_package(Qt5 COMPONENTS Core Widgets Gui PrintSupport OpenGL REQUIRED)
qt5_wrap_cpp (FOO_MOC_OUTFILES ${FOO_MOC_HEADERS})
message("MOC: ${FOO_MOC_OUTFILES}")

add_library(${targetname} SHARED ${header_list} ${source_list} ${FOO_MOC_OUTFILES})
set_target_properties(${targetname} PROPERTIES OUTPUT_NAME "${targetname}$<$<CONFIG:Debug>:d>")

set(youroutputdirectory "${CMAKE_SOURCE_DIR}/../bin/plugins")
set_target_properties( ${targetname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${youroutputdirectory} )
set_target_properties( ${targetname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${youroutputdirectory}/Debug" )
set_target_properties( ${targetname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${youroutputdirectory} )
	
set_property(TARGET ${targetname} PROPERTY CXX_STANDARD 17)

target_link_libraries(${targetname} Qt5::Widgets Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::PrintSupport Qt5::WinMain)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
target_link_libraries(${targetname} ${OPENGL_LIBRARIES})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})
target_link_libraries(${targetname} ${GLEW_STATIC_LIBRARIES})

find_package(GLM REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")
include_directories(${GLM_INCLUDE_DIR})

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

target_link_libraries(${targetname} debug poca_cored.lib)
target_link_libraries(${targetname} optimized poca_core.lib)
target_link_libraries(${targetname} debug poca_geometryd.lib)
target_link_libraries(${targetname} optimized poca_geometry.lib)
target_link_libraries(${targetname} debug poca_opengld.lib)
target_link_libraries(${targetname} optimized poca_opengl.lib)
target_link_libraries(${targetname} debug poca_plotd.lib)
target_link_libraries(${targetname} optimized poca_plot.lib)
