include(FindPackageHandleStandardArgs)

find_path(GLM_INCLUDE_DIR
    NAMES glm/glm.hpp
    HINTS ${GLM_ROOT} ${GLM_ROOT_DIR}
    PATH_SUFFIXES include)

set(GLM_INCLUDE_DIRS "${GLM_INCLUDE_DIR}")
find_package_handle_standard_args(GLM REQUIRED_VARS GLM_INCLUDE_DIR)

if(GLM_FOUND AND NOT TARGET GLM::GLM)
    add_library(GLM::GLM INTERFACE IMPORTED)
    set_target_properties(GLM::GLM PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIR}")
endif()

mark_as_advanced(GLM_INCLUDE_DIR)
