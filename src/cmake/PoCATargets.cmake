include(CMakeParseArguments)

function(poca_apply_common_target_settings target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE NOMINMAX QT_NO_WARNING_OUTPUT)
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "d")

    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX>:/W3>
            $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)

        # CUDA 13.x / CCCL requires the conforming MSVC preprocessor too.
        if(POCA_HAS_CUDA)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
        endif()

        # Qt 5.15.2 uses stdext::make_checked_array_iterator, which was removed
        # from MSVC 14.51 (VS 2026).  Force-include a tiny compatibility shim
        # only for that newer toolset.  Remove this when PoCA moves to Qt 6.
        if(MSVC_VERSION GREATER_EQUAL 1950)
            set(_poca_qt5_msvc_compat "${CMAKE_SOURCE_DIR}/cmake/Qt5Msvc2026Compat.hpp")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/FI${_poca_qt5_msvc_compat}>
                $<$<COMPILE_LANGUAGE:CUDA>:--pre-include=${_poca_qt5_msvc_compat}>)
        endif()
    endif()
endfunction()

function(poca_enable_cuda_for_target target)
    set(options RESOLVE_DEVICE_SYMBOLS)
    set(oneValueArgs)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(CUDA_ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(POCA_HAS_CUDA)
        if(CUDA_ARG_SOURCES)
            target_sources(${target} PRIVATE ${CUDA_ARG_SOURCES})
            set_source_files_properties(${CUDA_ARG_SOURCES} PROPERTIES LANGUAGE CUDA)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            # CUDA 13 changed the default linkage of host stubs generated for
            # __global__ function templates in whole-program mode (-rdc=false).
            # PoCA has several template kernels declared in headers and defined
            # in another CUDA translation unit, so keep the pre-CUDA-13 linkage
            # behavior unless/until those kernels are reorganized.
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
                target_compile_options(${target} PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:--static-global-template-stub=false>)
            endif()
        endif()
        target_link_libraries(${target} PRIVATE CUDA::cudart_static)

        # CUDA 13 moved Thrust/CUB/libcudacxx under include/cccl. Some PoCA
        # public headers (notably poca_core/Cuda/CoreMisc.h) include Thrust,
        # so consumers compiled as normal C++ also need these header paths.
        target_include_directories(${target} PUBLIC
            "${CUDAToolkit_INCLUDE_DIRS}"
            "${CUDAToolkit_INCLUDE_DIRS}/cccl")

        if(CUDA_ARG_RESOLVE_DEVICE_SYMBOLS)
            set_target_properties(${target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        endif()
    else()
        target_compile_definitions(${target} PRIVATE NO_CUDA)
    endif()
endfunction()

function(poca_group_sources target)
    get_target_property(_sources ${target} SOURCES)
    if(_sources)
        foreach(_source IN LISTS _sources)
            if(NOT _source MATCHES "^\\$<")
                if(IS_ABSOLUTE "${_source}")
                    set(_abs "${_source}")
                else()
                    set(_abs "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
                endif()
                if(EXISTS "${_abs}")
                    file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_abs}")
                    get_filename_component(_dir "${_rel}" DIRECTORY)
                    if(_dir)
                        string(REPLACE "/" "\\\\" _group "${_dir}")
                        source_group("${_group}" FILES "${_source}")
                    endif()
                endif()
            endif()
        endforeach()
    endif()
endfunction()

function(poca_add_static_library target)
    set(options USE_CUDA CUDA_RESOLVE_DEVICE_SYMBOLS)
    set(oneValueArgs ALIAS)
    set(multiValueArgs SOURCES CUDA_SOURCES PUBLIC_LINK_LIBRARIES PRIVATE_LINK_LIBRARIES PUBLIC_INCLUDE_DIRECTORIES PRIVATE_INCLUDE_DIRECTORIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_library(${target} STATIC ${ARG_SOURCES})
    poca_apply_common_target_settings(${target})
    target_include_directories(${target}
        PUBLIC  ${ARG_PUBLIC_INCLUDE_DIRECTORIES}
        PRIVATE ${ARG_PRIVATE_INCLUDE_DIRECTORIES})
    target_link_libraries(${target}
        PUBLIC  ${ARG_PUBLIC_LINK_LIBRARIES}
        PRIVATE ${ARG_PRIVATE_LINK_LIBRARIES})

    if(ARG_USE_CUDA)
        if(ARG_CUDA_RESOLVE_DEVICE_SYMBOLS)
            poca_enable_cuda_for_target(${target} SOURCES ${ARG_CUDA_SOURCES} RESOLVE_DEVICE_SYMBOLS)
        else()
            poca_enable_cuda_for_target(${target} SOURCES ${ARG_CUDA_SOURCES})
        endif()
    endif()

    set_target_properties(${target} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${POCA_LIBRARY_OUTPUT_DIR}/$<CONFIG>")

    if(ARG_ALIAS)
        add_library(${ARG_ALIAS} ALIAS ${target})
    endif()
    poca_group_sources(${target})
endfunction()

function(poca_add_plugin target)
    set(options USE_CUDA CUDA_RESOLVE_DEVICE_SYMBOLS)
    set(oneValueArgs)
    set(multiValueArgs SOURCES CUDA_SOURCES LINK_LIBRARIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_library(${target} MODULE ${ARG_SOURCES})
    poca_apply_common_target_settings(${target})
    target_include_directories(${target} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
    target_link_libraries(${target} PRIVATE PoCA::PluginAPI ${ARG_LINK_LIBRARIES})

    if(ARG_USE_CUDA)
        if(ARG_CUDA_RESOLVE_DEVICE_SYMBOLS)
            poca_enable_cuda_for_target(${target} SOURCES ${ARG_CUDA_SOURCES} RESOLVE_DEVICE_SYMBOLS)
        else()
            poca_enable_cuda_for_target(${target} SOURCES ${ARG_CUDA_SOURCES})
        endif()
    endif()

    # Keep the legacy runtime layout expected by Engine::loadPlugin():
    #   Debug   -> bin/plugins/Debug/*d.dll
    #   Release -> bin/plugins/*.dll
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_DEBUG          "${POCA_PLUGIN_OUTPUT_DIR}/Debug"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE        "${POCA_PLUGIN_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${POCA_PLUGIN_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     "${POCA_PLUGIN_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${POCA_PLUGIN_OUTPUT_DIR}/Debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${POCA_PLUGIN_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${POCA_PLUGIN_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${POCA_PLUGIN_OUTPUT_DIR}")
    poca_group_sources(${target})
endfunction()

function(poca_add_executable target)
    set(options USE_CUDA)
    set(oneValueArgs)
    set(multiValueArgs SOURCES LINK_LIBRARIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${target} ${ARG_SOURCES})
    poca_apply_common_target_settings(${target})
    target_include_directories(${target} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
    target_link_libraries(${target} PRIVATE PoCA::PluginAPI ${ARG_LINK_LIBRARIES})
    if(ARG_USE_CUDA)
        poca_enable_cuda_for_target(${target})
    endif()
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${POCA_RUNTIME_OUTPUT_DIR}/$<CONFIG>")
    poca_group_sources(${target})
endfunction()
