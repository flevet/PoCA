set(targetname "poca_core")
message("***************** Processing ${targetname}")

project(${targetname})

include_directories(${CMAKE_SOURCE_DIR}/poca_opengl)
link_directories(${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

#cmake_minimum_required(VERSION 3.6)

#if(POLICY CMP0020)
#	cmake_policy(SET CMP0020 OLD)
#endif()

#if(COMMAND cmake_policy)
#	cmake_policy(SET CMP0003 NEW)
#endif(COMMAND cmake_policy)


#set(CMAKE_AUTOMOC ON)
#set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#if(WIN32)
#  set(CMAKE_WIN32_EXECUTABLE ON)
#endif()

# This code sorts the project files as they appear in the root directory

# Generate a list of all .c & .h files in the current directory and sub directores.

set(MISC_HEADER_FILES
  Cuda/CoreMisc.h
  Cuda/ConnectedComponents.h
  )

set(DESIGN_PATTERNS_HEADER_FILES
  DesignPatterns/ChangeManagerSingleton.hpp
  DesignPatterns/MacroRecorderSingleton.hpp
  DesignPatterns/MediatorWObjectFWidget.hpp
  DesignPatterns/Observer.hpp
  DesignPatterns/Subject.hpp
  )

set(DESIGN_PATTERNS_SOURCE_FILES
  DesignPatterns/ChangeManagerSingleton.cpp
  DesignPatterns/MacroRecorderSingleton.cpp
  DesignPatterns/MediatorWObjectFWidget.cpp
  DesignPatterns/Subject.cpp
)

set(FIT_HEADER_FILES
  Fit/lmcurve.h
  Fit/lmmin.h
  Fit/lmstruct.h
  )

set(FIT_SOURCE_FILES
  Fit/lmcurve.c
  Fit/lmmin.c
)

set(GENERAL_HEADER_FILES
  General/ArrayStatistics.hpp
  General/BasicComponent.hpp
  General/BasicComponentList.hpp
  General/Command.hpp
  General/CommandableObject.hpp
  General/EquationFit.hpp
  General/Engine.hpp
  General/Histogram.hpp
  General/Image.hpp
  General/json.hpp
  General/Misc.h
  General/MyArray.hpp
  General/MyData.hpp
  General/Palette.hpp
  General/PluginList.hpp
  General/PythonInterpreter.hpp
  General/ranker.h
  General/Scatterplot.hpp
  General/Vec2.hpp
  General/Vec3.hpp
  General/Vec4.hpp
  General/Vec6.hpp
  )
  
  set(GENERAL_SOURCE_FILES
  General/ArrayStatistics.cpp
  General/BasicComponent.cpp
  General/CommandableObject.cpp
  General/EquationFit.cpp
  General/Engine.cpp
  General/Misc.cpp
  General/MyData.cpp
  General/Palette.cpp
  General/PluginList.cpp
  General/PythonInterpreter.cpp
  General/Scatterplot.cpp
)

set(INTERFACES_HEADER_FILES
  Interfaces/BasicComponentInterface.hpp
  Interfaces/CommandableObjectInterface.hpp
  Interfaces/HistogramInterface.hpp
  Interfaces/ImageInterface.hpp
  Interfaces/MyObjectInterface.hpp
  Interfaces/PaletteInterface.hpp
  Interfaces/ROIInterface.hpp
  Interfaces/SubjectInterface.hpp
  Interfaces/ScatterplotInterface.hpp
  )
  
set(MY_OBJECT_HEADER_FILES
  Objects/MyObject.hpp
  Objects/MyMultipleObject.hpp
  Objects/MyObjectDisplayCommand.hpp
  )

set(MY_OBJECT_SOURCE_FILES
  Objects/MyObject.cpp
  Objects/MyMultipleObject.cpp
  Objects/MyObjectDisplayCommand.cpp
  )

set(header_list
  ${MISC_HEADER_FILES}
  ${DESIGN_PATTERNS_HEADER_FILES}
  ${FIT_HEADER_FILES}
  ${GENERAL_HEADER_FILES}
  ${INTERFACES_HEADER_FILES}
  ${MY_OBJECT_HEADER_FILES}
  )

set(source_list
  ${DESIGN_PATTERNS_SOURCE_FILES}
  ${FIT_SOURCE_FILES}
  ${GENERAL_SOURCE_FILES}
  ${MY_OBJECT_SOURCE_FILES}
  )

set(cuda_list
  Cuda/CoreMisc.cu
  Cuda/ConnectedComponents.cu
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

find_package(CUDA QUIET)

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
#add_library(${targetname} STATIC ${header_list} ${source_list})

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

include_directories(${Boost_INCLUDE_DIR})

#set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${targetname})
set_target_properties(${targetname} PROPERTIES OUTPUT_NAME "${targetname}$<$<CONFIG:Debug>:d>")

set(youroutputdirectory "${CMAKE_SOURCE_DIR}/../bin/lib")
set_target_properties( ${targetname} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${youroutputdirectory} )
set_target_properties( ${targetname} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${youroutputdirectory} )
set_target_properties( ${targetname} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${youroutputdirectory} )

set_property(TARGET ${targetname} PROPERTY CXX_STANDARD 17)

target_link_libraries(${targetname} Qt5::Core Qt5::Widgets)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
target_link_libraries(${targetname} ${OPENGL_LIBRARIES})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})
target_link_libraries(${targetname} ${GLEW_STATIC_LIBRARIES})

find_package(GLM REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")
include_directories(${GLM_INCLUDE_DIR})

find_package(TinyTIFF)
target_link_libraries(${targetname} TinyTIFF::TinyTIFF)

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
	  #target_link_libraries(${targetname} ${PYTHON_LIBRARIES})
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
