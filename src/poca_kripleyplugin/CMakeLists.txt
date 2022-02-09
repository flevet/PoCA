#if(WIN32)
#  set(CMAKE_WIN32_EXECUTABLE ON)
#endif()

# This code sorts the project files as they appear in the root directory

# Generate a list of all .c & .h files in the current directory and sub directores.

set(targetname "poca_kripleyplugin")
message("***************** Processing ${targetname}")

project(${targetname})

include_directories(${CMAKE_SOURCE_DIR}/poca_core)
include_directories(${CMAKE_SOURCE_DIR}/poca_opengl)
include_directories(${CMAKE_SOURCE_DIR}/poca_plot)
include_directories(${CMAKE_SOURCE_DIR}/poca_geometry)
include_directories(${CMAKE_SOURCE_DIR}/../include)
link_directories(${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

set(KRIPKEY_PLUGIN_HEADER_FILES
  KRipleyPlugin.hpp
  KRipleyWidget.hpp
  KRipley.hpp
  )

set(KRIPKEY_SOURCE_HEADER_FILES
  KRipleyPlugin.cpp
  KRipleyWidget.cpp
  KRipley.cpp
)

set(FOO_MOC_HEADERS
  KRipleyPlugin.hpp
  KRipleyWidget.hpp
)

set(FOO_SOURCE_FILES
  KRipleyPlugin.cpp
  KRipleyWidget.cpp
)

set(header_list
  ${KRIPKEY_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${KRIPKEY_SOURCE_HEADER_FILES}
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

target_link_libraries(${targetname} debug poca_cored.lib)
target_link_libraries(${targetname} optimized poca_core.lib)
target_link_libraries(${targetname} debug poca_geometryd.lib)
target_link_libraries(${targetname} optimized poca_geometry.lib)
target_link_libraries(${targetname} debug poca_opengld.lib)
target_link_libraries(${targetname} optimized poca_opengl.lib)
target_link_libraries(${targetname} debug poca_plotd.lib)
target_link_libraries(${targetname} optimized poca_plot.lib)
