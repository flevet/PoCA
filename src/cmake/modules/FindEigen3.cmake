# Find Eigen3 from either an installed tree or a header-only source tree.
#
# Result variables:
#   Eigen3_FOUND
#   EIGEN3_FOUND
#   Eigen3_VERSION / EIGEN3_VERSION
#   Eigen3_INCLUDE_DIRS / EIGEN3_INCLUDE_DIRS
#   EIGEN3_INCLUDE_DIR
#
# Imported target:
#   Eigen3::Eigen
#
# Optional hints:
#   Eigen3_ROOT, EIGEN3_ROOT and their environment-variable equivalents.

include(FindPackageHandleStandardArgs)

set(_eigen3_hints)
foreach(_root_var Eigen3_ROOT EIGEN3_ROOT EIGEN3_ROOT_DIR)
    if(DEFINED ${_root_var} AND NOT "${${_root_var}}" STREQUAL "")
        list(APPEND _eigen3_hints "${${_root_var}}")
    endif()
    if(DEFINED ENV{${_root_var}} AND NOT "$ENV{${_root_var}}" STREQUAL "")
        list(APPEND _eigen3_hints "$ENV{${_root_var}}")
    endif()
endforeach()

# Eigen is header-only.  A source checkout is already a perfectly valid include
# directory because it contains Eigen/Core directly; an installed Eigen tree is
# normally found below include/eigen3.
find_path(EIGEN3_INCLUDE_DIR
    NAMES Eigen/Core
    HINTS ${_eigen3_hints}
    PATH_SUFFIXES
        ""
        include
        include/eigen3
        eigen3
        eigen
)

unset(_eigen3_hints)

set(Eigen3_VERSION "")
if(EIGEN3_INCLUDE_DIR)
    set(_eigen3_macros "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h")
    if(EXISTS "${_eigen3_macros}")
        file(STRINGS "${_eigen3_macros}" _eigen3_world_line REGEX "^[ \t]*#define[ \t]+EIGEN_WORLD_VERSION[ \t]+[0-9]+")
        file(STRINGS "${_eigen3_macros}" _eigen3_major_line REGEX "^[ \t]*#define[ \t]+EIGEN_MAJOR_VERSION[ \t]+[0-9]+")
        file(STRINGS "${_eigen3_macros}" _eigen3_minor_line REGEX "^[ \t]*#define[ \t]+EIGEN_MINOR_VERSION[ \t]+[0-9]+")

        string(REGEX REPLACE ".*EIGEN_WORLD_VERSION[ \t]+([0-9]+).*" "\\1" _eigen3_world "${_eigen3_world_line}")
        string(REGEX REPLACE ".*EIGEN_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1" _eigen3_major "${_eigen3_major_line}")
        string(REGEX REPLACE ".*EIGEN_MINOR_VERSION[ \t]+([0-9]+).*" "\\1" _eigen3_minor "${_eigen3_minor_line}")

        if(_eigen3_world MATCHES "^[0-9]+$" AND
           _eigen3_major MATCHES "^[0-9]+$" AND
           _eigen3_minor MATCHES "^[0-9]+$")
            set(Eigen3_VERSION "${_eigen3_world}.${_eigen3_major}.${_eigen3_minor}")
        endif()

        unset(_eigen3_world_line)
        unset(_eigen3_major_line)
        unset(_eigen3_minor_line)
        unset(_eigen3_world)
        unset(_eigen3_major)
        unset(_eigen3_minor)
        unset(_eigen3_macros)
    endif()
endif()

find_package_handle_standard_args(Eigen3
    REQUIRED_VARS EIGEN3_INCLUDE_DIR
    VERSION_VAR Eigen3_VERSION
)

if(Eigen3_FOUND)
    set(EIGEN3_FOUND TRUE)
    set(EIGEN3_VERSION "${Eigen3_VERSION}")
    set(Eigen3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
    set(EIGEN3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")

    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    endif()
else()
    set(EIGEN3_FOUND FALSE)
endif()

mark_as_advanced(EIGEN3_INCLUDE_DIR)
