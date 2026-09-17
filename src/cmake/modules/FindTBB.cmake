include(FindPackageHandleStandardArgs)

set(_TBB_HINTS ${TBB_ROOT} ${TBB_ROOT_DIR} ${TBB_DIR} $ENV{TBBROOT})
find_path(TBB_INCLUDE_DIR NAMES tbb/tbb.h HINTS ${_TBB_HINTS} PATH_SUFFIXES include)

if(MSVC)
    set(_TBB_LIB_SUFFIXES lib/intel64/vc14 lib/intel64/vc15 lib/intel64/vc14_uwp lib)
else()
    set(_TBB_LIB_SUFFIXES lib lib64 lib/intel64/gcc4.8 lib/intel64/gcc4.7 lib/intel64/gcc4.4)
endif()

find_library(TBB_tbb_LIBRARY_RELEASE NAMES tbb HINTS ${_TBB_HINTS} PATH_SUFFIXES ${_TBB_LIB_SUFFIXES})
find_library(TBB_tbb_LIBRARY_DEBUG   NAMES tbb_debug HINTS ${_TBB_HINTS} PATH_SUFFIXES ${_TBB_LIB_SUFFIXES})
find_library(TBB_tbbmalloc_LIBRARY_RELEASE NAMES tbbmalloc HINTS ${_TBB_HINTS} PATH_SUFFIXES ${_TBB_LIB_SUFFIXES})
find_library(TBB_tbbmalloc_LIBRARY_DEBUG   NAMES tbbmalloc_debug HINTS ${_TBB_HINTS} PATH_SUFFIXES ${_TBB_LIB_SUFFIXES})

set(TBB_INCLUDE_DIRS "${TBB_INCLUDE_DIR}")
set(TBB_tbb_FOUND FALSE)
set(TBB_tbbmalloc_FOUND FALSE)
if(TBB_tbb_LIBRARY_RELEASE OR TBB_tbb_LIBRARY_DEBUG)
    set(TBB_tbb_FOUND TRUE)
endif()
if(TBB_tbbmalloc_LIBRARY_RELEASE OR TBB_tbbmalloc_LIBRARY_DEBUG)
    set(TBB_tbbmalloc_FOUND TRUE)
endif()
find_package_handle_standard_args(TBB
    REQUIRED_VARS TBB_INCLUDE_DIR TBB_tbb_LIBRARY_RELEASE TBB_tbbmalloc_LIBRARY_RELEASE
    HANDLE_COMPONENTS)

function(_poca_define_tbb_target component release_lib debug_lib)
    if(NOT TARGET TBB::${component})
        add_library(TBB::${component} UNKNOWN IMPORTED)
        set_target_properties(TBB::${component} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}")
        if(release_lib)
            set_property(TARGET TBB::${component} PROPERTY IMPORTED_LOCATION_RELEASE "${release_lib}")
            set_property(TARGET TBB::${component} PROPERTY IMPORTED_LOCATION_RELWITHDEBINFO "${release_lib}")
            set_property(TARGET TBB::${component} PROPERTY IMPORTED_LOCATION_MINSIZEREL "${release_lib}")
        endif()
        if(debug_lib)
            set_property(TARGET TBB::${component} PROPERTY IMPORTED_LOCATION_DEBUG "${debug_lib}")
            set_property(TARGET TBB::${component} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "$<$<CONFIG:Debug>:TBB_USE_DEBUG=1>")
        elseif(release_lib)
            set_property(TARGET TBB::${component} PROPERTY IMPORTED_LOCATION_DEBUG "${release_lib}")
        endif()
        set_property(TARGET TBB::${component} PROPERTY IMPORTED_CONFIGURATIONS "Debug;Release;RelWithDebInfo;MinSizeRel")
    endif()
endfunction()

if(TBB_FOUND)
    _poca_define_tbb_target(tbb "${TBB_tbb_LIBRARY_RELEASE}" "${TBB_tbb_LIBRARY_DEBUG}")
    _poca_define_tbb_target(tbbmalloc "${TBB_tbbmalloc_LIBRARY_RELEASE}" "${TBB_tbbmalloc_LIBRARY_DEBUG}")
endif()

mark_as_advanced(TBB_INCLUDE_DIR TBB_tbb_LIBRARY_RELEASE TBB_tbb_LIBRARY_DEBUG
                 TBB_tbbmalloc_LIBRARY_RELEASE TBB_tbbmalloc_LIBRARY_DEBUG)
