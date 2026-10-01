# Compatibility shim for PoCA's legacy find_package(CUDA) calls.
#
# CMake's historical FindCUDA module was removed from the default module set
# for modern policy configurations. PoCA only relies on a small subset of the
# variables it provided, while CUDA compilation itself is already handled by
# CMake's native CUDA language support (enable_language(CUDA)).
#
# This module keeps the existing PoCA CMakeLists.txt files unchanged and maps
# the modern FindCUDAToolkit result back to the legacy variable names used by
# the project.

if(CMAKE_VERSION VERSION_LESS 3.17)
    # FindCUDAToolkit did not exist before CMake 3.17. On those older CMake
    # versions, fall back to the built-in legacy module.
    include("${CMAKE_ROOT}/Modules/FindCUDA.cmake")
    return()
endif()

set(_poca_cuda_find_args)
if(CUDA_FIND_QUIETLY)
    list(APPEND _poca_cuda_find_args QUIET)
endif()
if(CUDA_FIND_REQUIRED)
    list(APPEND _poca_cuda_find_args REQUIRED)
endif()
if(CUDA_FIND_VERSION_EXACT)
    list(APPEND _poca_cuda_find_args EXACT)
endif()

if(CUDA_FIND_VERSION)
    find_package(CUDAToolkit ${CUDA_FIND_VERSION} ${_poca_cuda_find_args})
else()
    find_package(CUDAToolkit ${_poca_cuda_find_args})
endif()

set(CUDA_FOUND ${CUDAToolkit_FOUND})

if(CUDAToolkit_FOUND)
    # Legacy FindCUDA result variables used by PoCA.
    set(CUDA_VERSION "${CUDAToolkit_VERSION}")
    set(CUDA_VERSION_STRING "${CUDAToolkit_VERSION}")
    set(CUDA_VERSION_MAJOR "${CUDAToolkit_VERSION_MAJOR}")
    set(CUDA_VERSION_MINOR "${CUDAToolkit_VERSION_MINOR}")
    set(CUDA_VERSION_PATCH "${CUDAToolkit_VERSION_PATCH}")

    set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_TARGET_DIR}")
    set(CUDA_TOOLKIT_INCLUDE "${CUDAToolkit_INCLUDE_DIRS}")
    set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    set(CUDA_NVCC_EXECUTABLE "${CUDAToolkit_NVCC_EXECUTABLE}")

    # PoCA links this variable in many targets. Imported targets are accepted
    # by target_link_libraries(), so preserving the variable name avoids edits
    # in every plugin CMakeLists.txt.
    set(CUDA_CUDART_LIBRARY CUDA::cudart)
    set(CUDA_cudart_LIBRARY CUDA::cudart)
    set(CUDA_cudart_static_LIBRARY CUDA::cudart_static)
endif()

unset(_poca_cuda_find_args)
