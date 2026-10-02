include_guard(GLOBAL)

option(POCA_ENABLE_ZARR "Enable local OME-Zarr loading using an installed poca_zarr_backend" OFF)
set(POCA_ZARR_BACKEND_ROOT "" CACHE PATH "Installed Release poca_zarr_backend root (include, lib, bin)")
set(POCA_ZARR_BACKEND_DEBUG_ROOT "" CACHE PATH "Optional installed Debug poca_zarr_backend root (include, lib, bin); if empty, <POCA_ZARR_BACKEND_ROOT>_debug is detected automatically")
if(NOT POCA_ENABLE_ZARR)
    return()
endif()
if(NOT POCA_BUILD_EXTRA)
    message(FATAL_ERROR "POCA_ENABLE_ZARR requires POCA_BUILD_EXTRA=ON for the Zarr loader.")
endif()

# Validate before PoCA's CUDA/dependency discovery. Never configure source dependencies.
# Release/RelWithDebInfo/MinSizeRel use POCA_ZARR_BACKEND_ROOT. Debug uses the
# optional POCA_ZARR_BACKEND_DEBUG_ROOT, or automatically discovers the sibling
# <POCA_ZARR_BACKEND_ROOT>_debug install. If no Debug backend is available, Debug
# falls back to the Release backend (safe at the C ABI boundary used here).
cmake_path(NORMAL_PATH POCA_ZARR_BACKEND_ROOT OUTPUT_VARIABLE _poca_zarr_release_root)
if(NOT _poca_zarr_release_root)
    message(FATAL_ERROR
        "PoCA Zarr support requested but POCA_ZARR_BACKEND_ROOT is empty. "
        "Set it to the installed Release backend directory.")
endif()

function(_poca_zarr_backend_paths _root _prefix)
    set(${_prefix}_header "${_root}/include/poca_zarr_backend.h" PARENT_SCOPE)
    if(WIN32)
        set(${_prefix}_library "${_root}/lib/poca_zarr_backend.lib" PARENT_SCOPE)
        set(${_prefix}_runtime "${_root}/bin/poca_zarr_backend.dll" PARENT_SCOPE)
    else()
        set(_library "${_root}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}poca_zarr_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(${_prefix}_library "${_library}" PARENT_SCOPE)
        set(${_prefix}_runtime "${_library}" PARENT_SCOPE)
    endif()
endfunction()

function(_poca_zarr_validate_backend _root _label _required)
    _poca_zarr_backend_paths("${_root}" _candidate)
    foreach(_file IN ITEMS "${_candidate_header}" "${_candidate_library}" "${_candidate_runtime}")
        if(NOT EXISTS "${_file}" OR IS_DIRECTORY "${_file}")
            if(_required)
                message(FATAL_ERROR
                    "PoCA Zarr ${_label} backend was not found. Root: ${_root}. Missing file: ${_file}")
            endif()
            set(POCA_ZARR_BACKEND_VALID FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(POCA_ZARR_BACKEND_VALID TRUE PARENT_SCOPE)
endfunction()

_poca_zarr_validate_backend("${_poca_zarr_release_root}" "Release" TRUE)
_poca_zarr_backend_paths("${_poca_zarr_release_root}" _poca_zarr_release)

set(_poca_zarr_debug_root "")
if(POCA_ZARR_BACKEND_DEBUG_ROOT)
    cmake_path(NORMAL_PATH POCA_ZARR_BACKEND_DEBUG_ROOT OUTPUT_VARIABLE _poca_zarr_debug_root)
else()
    set(_poca_zarr_debug_candidate "${_poca_zarr_release_root}_debug")
    _poca_zarr_validate_backend("${_poca_zarr_debug_candidate}" "auto-detected Debug" FALSE)
    if(POCA_ZARR_BACKEND_VALID)
        set(_poca_zarr_debug_root "${_poca_zarr_debug_candidate}")
    endif()
endif()

if(_poca_zarr_debug_root)
    _poca_zarr_validate_backend("${_poca_zarr_debug_root}" "Debug" TRUE)
    _poca_zarr_backend_paths("${_poca_zarr_debug_root}" _poca_zarr_debug)
    message(STATUS "PoCA Zarr backend: Release=${_poca_zarr_release_root}; Debug=${_poca_zarr_debug_root}")
else()
    set(_poca_zarr_debug_header "${_poca_zarr_release_header}")
    set(_poca_zarr_debug_library "${_poca_zarr_release_library}")
    set(_poca_zarr_debug_runtime "${_poca_zarr_release_runtime}")
    message(STATUS "PoCA Zarr backend: Release=${_poca_zarr_release_root}; Debug falls back to Release backend")
endif()

add_library(PoCA::ZarrBackend SHARED IMPORTED GLOBAL)
set_target_properties(PoCA::ZarrBackend PROPERTIES
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE;RELWITHDEBINFO;MINSIZEREL"
    IMPORTED_LOCATION "${_poca_zarr_release_runtime}"
    IMPORTED_LOCATION_DEBUG "${_poca_zarr_debug_runtime}"
    IMPORTED_LOCATION_RELEASE "${_poca_zarr_release_runtime}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${_poca_zarr_release_runtime}"
    IMPORTED_LOCATION_MINSIZEREL "${_poca_zarr_release_runtime}"
    INTERFACE_INCLUDE_DIRECTORIES "${_poca_zarr_release_root}/include")
if(WIN32)
    set_target_properties(PoCA::ZarrBackend PROPERTIES
        IMPORTED_IMPLIB "${_poca_zarr_release_library}"
        IMPORTED_IMPLIB_DEBUG "${_poca_zarr_debug_library}"
        IMPORTED_IMPLIB_RELEASE "${_poca_zarr_release_library}"
        IMPORTED_IMPLIB_RELWITHDEBINFO "${_poca_zarr_release_library}"
        IMPORTED_IMPLIB_MINSIZEREL "${_poca_zarr_release_library}")
endif()

unset(_poca_zarr_release_root)
unset(_poca_zarr_debug_root)
unset(_poca_zarr_debug_candidate)
unset(_poca_zarr_release_header)
unset(_poca_zarr_release_library)
unset(_poca_zarr_release_runtime)
unset(_poca_zarr_debug_header)
unset(_poca_zarr_debug_library)
unset(_poca_zarr_debug_runtime)
unset(_candidate_header)
unset(_candidate_library)
unset(_candidate_runtime)
unset(_library)
unset(POCA_ZARR_BACKEND_VALID)
