set(targetname "poca")
message("***************** Processing ${targetname}")

project(${targetname} LANGUAGES CXX)# CUDA)

include_directories(${CMAKE_SOURCE_DIR}/poca_core)
include_directories(${CMAKE_SOURCE_DIR}/poca_opengl)
include_directories(${CMAKE_SOURCE_DIR}/poca_plot)
include_directories(${CMAKE_SOURCE_DIR}/poca_geometry)
include_directories(${CMAKE_SOURCE_DIR}/../include)
link_directories(${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

set(SMLM_OBJECT_HEADER_FILES
  Objects/SMLM_Object/SMLMObject.hpp
  )

set(SMLM_OBJECT_SOURCE_FILES
  Objects/SMLM_Object/SMLMObject.cpp
  )

set(COLOC_OBJECT_HEADER_FILES
  Objects/Coloc_Object/ColocObject.hpp
  )

set(COLOC_OBJECT_SOURCE_FILES
  Objects/Coloc_Object/ColocObject.cpp
  )

set(WIDGET_HEADER_FILES
  Widgets/MainFilterWidget.hpp
  Widgets/MdiChild.hpp
  Widgets/ColocalizationChoiceDialog.hpp
  Widgets/MacroWidget.hpp
  Widgets/MainWindow.hpp
  Widgets/PythonParametersDialog.hpp
  Widgets/PythonWidget.hpp
  Widgets/ROIGeneralWidget.hpp
  )

set(WIDGET_SOURCE_FILES
  Widgets/MainFilterWidget.cpp
  Widgets/MdiChild.cpp
  Widgets/ColocalizationChoiceDialog.cpp
  Widgets/MacroWidget.cpp
  Widgets/MainWindow.cpp
  Widgets/PythonParametersDialog.cpp
  Widgets/PythonWidget.cpp
  Widgets/ROIGeneralWidget.cpp
 )

if(ADD_PYTHON)
	set(FOO_MOC_HEADERS
	  Widgets/MainFilterWidget.hpp
	  Widgets/MdiChild.hpp
	  Widgets/ColocalizationChoiceDialog.hpp
	  Widgets/MacroWidget.hpp
	  Widgets/MainWindow.hpp
	  Widgets/PythonParametersDialog.hpp
	  Widgets/PythonWidget.hpp
	  Widgets/ROIGeneralWidget.hpp
	)
else()
	set(FOO_MOC_HEADERS
	  Widgets/MainFilterWidget.hpp
	  Widgets/MdiChild.hpp
	  Widgets/ColocalizationChoiceDialog.hpp
	  Widgets/MacroWidget.hpp
	  Widgets/MainWindow.hpp
	  Widgets/ROIGeneralWidget.hpp
	  Widgets/PythonParametersDialog.hpp
	)

endif()

if(ADD_PYTHON)
	set(FOO_SOURCE_FILES
	  Widgets/MainFilterWidget.cpp
	  Widgets/MdiChild.cpp
	  Widgets/ColocalizationChoiceDialog.cpp
	  Widgets/MacroWidget.cpp
	  Widgets/MainWindow.cpp
	  Widgets/PythonParametersDialog.cpp
	  Widgets/PythonWidget.cpp
	  Widgets/ROIGeneralWidget.cpp
	  )
else()
	set(FOO_SOURCE_FILES
	  Widgets/MainFilterWidget.cpp
	  Widgets/MdiChild.cpp
	  Widgets/ColocalizationChoiceDialog.cpp
	  Widgets/MacroWidget.cpp
	  Widgets/MainWindow.cpp
	  Widgets/ROIGeneralWidget.cpp
	  Widgets/PythonParametersDialog.cpp
	  )

endif()

set(header_list
  ${SMLM_OBJECT_HEADER_FILES}
  ${COLOC_OBJECT_HEADER_FILES}
  ${WIDGET_HEADER_FILES}
  )

message("Headers: ${RESOURCE_FILES}")

set(source_list
  ${SMLM_OBJECT_SOURCE_FILES}
  ${COLOC_OBJECT_SOURCE_FILES}
  ${WIDGET_SOURCE_FILES}
  )

set(main_CPP
  main.cpp
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
get_filename_component(main_path "${main_CPP}" PATH)
string(REPLACE "/" "\\" main_path_msvc "${main_path}")
source_group("${main_path_msvc}" FILES "${main_CPP}")

add_definitions(-DNOMINMAX)
add_definitions(-DQT_NO_WARNING_OUTPUT)

find_package(CUDA QUIET)

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

find_package(Qt5 COMPONENTS Core Widgets Gui PrintSupport OpenGL REQUIRED)
qt5_wrap_cpp (FOO_MOC_OUTFILES ${FOO_MOC_HEADERS})
message("MOC: ${FOO_MOC_OUTFILES}")

add_executable(${targetname} ${header_list} ${source_list} ${main_CPP} ${FOO_MOC_OUTFILES})

if(CGAL_FOUND)
    message("Found CGAL.")
    if(CGAL_VERSION VERSION_GREATER_EQUAL "6.0")
	message("CGAL version 6")
	target_link_libraries(${targetname} CGAL::CGAL)
    endif()
endif()

if(CUDA_FOUND)
  message("CUDA was found")
  #set(CMAKE_CUDA_FLAGS "-Xcompiler=\"/EHsc -Zi -Ob0\"")
  message("CUDA flags: ${CMAKE_CUDA_FLAGS}")
  enable_language(CUDA)
  #set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${targetname} ${CUDA_cudart_static_LIBRARY})
#	foreach(source IN LISTS cuda_list)
#		get_filename_component(source_path "${source}" PATH)
#		string(REPLACE "/" "\\" source_path_msvc "${source_path}")
#		message(STATUS ${source_path_msvc})
#		source_group("${source_path_msvc}" FILES "${source}")
#	endforeach()
#	add_executable(poca ${header_list} ${source_list} ${cuda_list} ${main_CPP} ${FOO_MOC_OUTFILES})
	#target_compile_options(poca PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:"-Xcompiler="/EHsc -Zi -Ob0 -Ob2"">) 
else()
	message("CUDA not found")
	add_definitions(-DNO_CUDA)
#	add_executable(poca ${header_list} ${source_list} ${main_CPP} ${FOO_MOC_OUTFILES})
endif()

#set(CUDA_TARGET_INC C:/Program\ Files/NVIDIA GPU\ Computing\ Toolkit/CUDA/v11.1/include)
#set(CUDA_SAMPLE_LKN C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.1/lib/x64)
#include_directories(${CUDA_TARGET_INC})
#link_libraries(${CUDA_SAMPLE_LKN})
	
set_target_properties(${targetname} PROPERTIES OUTPUT_NAME "poca$<$<CONFIG:Debug>:d>")
set_property(TARGET ${targetname} PROPERTY CXX_STANDARD 17)
IF(MSVC)
	message("C++ flags: ${CMAKE_CXX_FLAGS}")
	message("cmake flags changed")
    #SET(CMAKE_CXX_FLAGS "/EHsc")
	set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /EHsc")
	message("C++ flags debug: ${CMAKE_CXX_FLAGS_DEBUG}")
ENDIF(MSVC)

target_link_libraries(${targetname} Qt5::Widgets Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::PrintSupport)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
target_link_libraries(${targetname} ${OPENGL_LIBRARIES})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})
target_link_libraries(${targetname} ${GLEW_STATIC_LIBRARIES})

find_package(GLM REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")
include_directories(${GLM_INCLUDE_DIR})

find_package(FFmpeg REQUIRED)
include_directories(
    ${AVCODEC_INCLUDE_DIR}
    ${AVFORMAT_INCLUDE_DIR}
    ${AVUTIL_INCLUDE_DIR}
    ${AVDEVICE_INCLUDE_DIR}
)

target_link_libraries(${targetname}
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${AVDEVICE_LIBRARY}
    ${SWSCALE_LIBRARY}
    ${SWRESAMPLE_LIBRARY}
)

find_package(direct-to-video REQUIRED)
include_directories(${direct-to-video_INCLUDE_DIRS})
#target_link_libraries(${targetname} ${direct-to-video_LIBRARIES})
target_link_libraries(${targetname} debug ${direct-to-video_LIBRARY_DEBUG})
target_link_libraries(${targetname} optimized ${direct-to-video_LIBRARY_RELEASE})

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

if(ADD_PYTHON)
	#Add Python
	find_package(PythonInterp)
	find_package(PythonLibs)
	set(HAS_PYTHON True)
	# require at least python 2.6
	if(PYTHON_VERSION_STRING VERSION_LESS 2.7)
	  message(WARNING "Python version less than 2.7: \"${PYTHON_VERSION_STRING}\".")
	  set(HAS_PYTHON False)
	endif()

	# check for version mismatch.
	if(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND
		AND NOT(PYTHON_VERSION_STRING VERSION_EQUAL PYTHONLIBS_VERSION_STRING))
	  message(WARNING "Python executable (\"${PYTHON_VERSION_STRING}\") and library (\"${PYTHONLIBS_VERSION_STRING}\") version mismatch.")
	  set(HAS_PYTHON False)
	endif()
	if(HAS_PYTHON)
	  include_directories(${PYTHON_INCLUDE_DIRS})
	  target_link_libraries(${targetname} ${PYTHON_LIBRARIES})
	  find_package(NUMPY)
	  if(PYTHON_NUMPY_FOUND)
	    include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
	  endif()
	else()
	  add_definitions(-DNO_PYTHON)
	endif()
else()
	add_definitions(-DNO_PYTHON)
endif()

target_link_libraries(${targetname} debug poca_cored.lib)
target_link_libraries(${targetname} optimized poca_core.lib)
target_link_libraries(${targetname} debug poca_geometryd.lib)
target_link_libraries(${targetname} optimized poca_geometry.lib)
target_link_libraries(${targetname} debug poca_opengld.lib)
target_link_libraries(${targetname} optimized poca_opengl.lib)
target_link_libraries(${targetname} debug poca_plotd.lib)
target_link_libraries(${targetname} optimized poca_plot.lib)