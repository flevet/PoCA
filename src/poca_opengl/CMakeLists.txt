#if(WIN32)
#  set(CMAKE_WIN32_EXECUTABLE ON)
#endif()

# This code sorts the project files as they appear in the root directory

# Generate a list of all .c & .h files in the current directory and sub directores.

set(targetname "poca_opengl")
message("***************** Processing ${targetname}")

project(${targetname})

include_directories(${CMAKE_SOURCE_DIR}/poca_core)
link_directories(${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

set(MISC_HEADER_FILES
  Cuda/Misc.h
  )
  
set(MISC_SOURCE_FILES
  Cuda/Misc.cpp
  )
  
set(OPENGL_HEADER_FILES
  OpenGL/BasicDisplayCommand.hpp
  OpenGL/Camera.hpp
  OpenGL/GLBuffer.hpp
  OpenGL/Helper.h
  OpenGL/SsaoShader.hpp
  OpenGL/Shader.hpp
  OpenGL/TextDisplayer.hpp
  )

set(OPENGL_SOURCE_FILES
  OpenGL/BasicDisplayCommand.cpp
  OpenGL/Camera.cpp
  OpenGL/Helper.cpp
  OpenGL/SsaoShader.cpp
  OpenGL/TextDisplayer.cpp
)

set(GENERAL_HEADER_FILES
  General/Roi.hpp
  )
  
set(GENERAL_SOURCE_FILES
  General/Roi.cpp
  )

set(PLOT_HEADER_FILES
  Plot/ScatterplotGL.hpp
   )

set(PLOT_SOURCE_FILES
  Plot/ScatterplotGL.cpp
)

set(INTERFACES_HEADER_FILES
  Interfaces/CameraInterface.hpp
  )
  
 set(FONTSTASH_HEADER_FILES
  fontstash/fontstash.h
  fontstash/gl3corefontstash.h
  fontstash/stb_truetype.h
  )
  
set(FOO_MOC_HEADERS
  OpenGL/Camera.hpp
  Plot/ScatterplotGL.hpp
)

set(FOO_SOURCE_FILES
  OpenGL/Camera.cpp
  Plot/ScatterplotGL.cpp
)

set(header_list
  ${MISC_HEADER_FILES}
  ${OPENGL_HEADER_FILES}
  ${GENERAL_HEADER_FILES}
  ${PLOT_HEADER_FILES}
  ${INTERFACES_HEADER_FILES}
  ${FONTSTASH_HEADER_FILES}
  )

set(source_list
  ${MISC_SOURCE_FILES}
  ${OPENGL_SOURCE_FILES}
  ${GENERAL_SOURCE_FILES}
  ${PLOT_SOURCE_FILES}
  )
  
set(cuda_list
  Cuda/Misc.cu
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
find_package(Qt5 COMPONENTS Core OpenGL REQUIRED)
qt5_wrap_cpp (FOO_MOC_OUTFILES ${FOO_MOC_HEADERS})
message("MOC: ${FOO_MOC_OUTFILES}")

find_package(CUDA QUIET)
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
	add_library(${targetname} STATIC ${header_list} ${source_list} ${cuda_list} ${FOO_MOC_OUTFILES})
	#target_compile_options(poca PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:"-Xcompiler="/EHsc -Zi -Ob0 -Ob2"">) 
else()
	message("CUDA not found")
	add_definitions(-DNO_CUDA)
	add_library(${targetname} STATIC ${header_list} ${source_list} ${FOO_MOC_OUTFILES})
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#add_library(${targetname} STATIC ${header_list} ${source_list} ${FOO_MOC_OUTFILES})
set_target_properties(${targetname} PROPERTIES OUTPUT_NAME "${targetname}$<$<CONFIG:Debug>:d>")
set_target_properties(${targetname} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(youroutputdirectory "${CMAKE_SOURCE_DIR}/../bin/lib")
set_target_properties( ${targetname} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${youroutputdirectory} )
set_target_properties( ${targetname} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${youroutputdirectory} )
set_target_properties( ${targetname} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${youroutputdirectory} )

set_property(TARGET ${targetname} PROPERTY CXX_STANDARD 17)

target_link_libraries(${targetname} Qt5::Core Qt5::OpenGL)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
target_link_libraries(${targetname} ${OPENGL_LIBRARIES})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})
target_link_libraries(${targetname} ${GLEW_STATIC_LIBRARIES})

find_package(GLM REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")
include_directories(${GLM_INCLUDE_DIR})
