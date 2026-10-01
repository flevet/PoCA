# Central third-party dependency discovery for PoCA.
# Dependencies are found once here; individual targets only state what they link to.

# Optional root hints exposed in CMake GUI. Leave them empty when the package is already discoverable.
set(GLEW_ROOT "" CACHE PATH "Root directory of the GLEW installation")
set(GLM_ROOT "" CACHE PATH "Root directory containing glm/glm.hpp")
set(TBB_ROOT "" CACHE PATH "Root directory of the Intel TBB installation")
set(Boost_ROOT "" CACHE PATH "Root directory of the Boost headers used by CGAL")
set(Eigen3_ROOT "" CACHE PATH "Eigen source/install root containing Eigen/Core, or an installation prefix")
set(GEMMI_ROOT "" CACHE PATH "Gemmi repository root or include directory")

# Preserve the Eigen path used by current main, but do not force it on a machine
# whose dependency layout differs. An explicit Eigen3_ROOT always wins.
if(NOT Eigen3_ROOT AND POCA_EIGEN5_DIR)
    set(Eigen3_ROOT "${POCA_EIGEN5_DIR}")
endif()

find_package(Qt5 5.15 REQUIRED COMPONENTS Core Gui Widgets OpenGL PrintSupport)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLM REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb tbbmalloc)

# Eigen must be discoverable before CGAL is configured. PoCA's custom finder
# accepts both a source checkout (Eigen/Core directly below the root) and an
# installed include/eigen3 layout.
find_package(Eigen3 3.1 REQUIRED)
if(NOT TARGET Eigen3::Eigen)
    if(EIGEN3_INCLUDE_DIRS)
        set(_poca_eigen_include_dirs "${EIGEN3_INCLUDE_DIRS}")
    elseif(EIGEN3_INCLUDE_DIR)
        set(_poca_eigen_include_dirs "${EIGEN3_INCLUDE_DIR}")
    elseif(Eigen3_INCLUDE_DIRS)
        set(_poca_eigen_include_dirs "${Eigen3_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR
            "Eigen3 was found, but no Eigen3::Eigen target or Eigen include directory was provided.")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_poca_eigen_include_dirs}")
    unset(_poca_eigen_include_dirs)
endif()
message(STATUS "PoCA Eigen: ${Eigen3_VERSION} (${EIGEN3_INCLUDE_DIR})")

# Support both the older CGAL package used by the VS 2022 setup and CGAL 6.x
# used by the newer setup. CMake 3.30+ needs CMP0167=OLD while older CGAL asks
# CMake to locate Boost through FindBoost.
if(POLICY CMP0167)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0167 OLD)
endif()
find_package(CGAL REQUIRED COMPONENTS Core)
if(POLICY CMP0167)
    cmake_policy(POP)
endif()
if(CGAL_VERSION VERSION_LESS "6.0" AND CGAL_USE_FILE AND EXISTS "${CGAL_USE_FILE}")
    include("${CGAL_USE_FILE}")
endif()
message(STATUS "PoCA CGAL: ${CGAL_VERSION}")

# The legacy machine and the modern machine can expose these packages either as
# CMake config packages or through classic Find modules. Normalize both cases
# to the imported target names used by the target-based build.
find_package(TinyTIFF CONFIG QUIET)
if(NOT TARGET TinyTIFF::TinyTIFF)
    find_package(TinyTIFF REQUIRED)
endif()
if(NOT TARGET TinyTIFF::TinyTIFF)
    add_library(TinyTIFF::TinyTIFF INTERFACE IMPORTED)
    if(TINYTIFF_INCLUDE_DIRS)
        set_property(TARGET TinyTIFF::TinyTIFF PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${TINYTIFF_INCLUDE_DIRS}")
    elseif(TinyTIFF_INCLUDE_DIRS)
        set_property(TARGET TinyTIFF::TinyTIFF PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${TinyTIFF_INCLUDE_DIRS}")
    endif()
    if(TINYTIFF_LIBRARIES)
        set_property(TARGET TinyTIFF::TinyTIFF PROPERTY INTERFACE_LINK_LIBRARIES "${TINYTIFF_LIBRARIES}")
    elseif(TinyTIFF_LIBRARIES)
        set_property(TARGET TinyTIFF::TinyTIFF PROPERTY INTERFACE_LINK_LIBRARIES "${TinyTIFF_LIBRARIES}")
    endif()
endif()

find_package(tinysplinecxx CONFIG QUIET)
if(NOT TARGET tinysplinecxx::tinysplinecxx)
    find_package(tinysplinecxx REQUIRED)
endif()
if(NOT TARGET tinysplinecxx::tinysplinecxx)
    add_library(tinysplinecxx::tinysplinecxx INTERFACE IMPORTED)
    if(TINYSPLINECXX_INCLUDE_DIRS)
        set_property(TARGET tinysplinecxx::tinysplinecxx PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES "${TINYSPLINECXX_INCLUDE_DIRS}")
    endif()
    if(TINYSPLINECXX_LIBRARIES)
        set_property(TARGET tinysplinecxx::tinysplinecxx PROPERTY
            INTERFACE_LINK_LIBRARIES "${TINYSPLINECXX_LIBRARIES}")
    endif()
endif()
find_package(OpenMP QUIET)

# Gemmi is header-only and is used by the PDB/mmCIF loader.
set(_poca_gemmi_hints)
if(GEMMI_ROOT)
    list(APPEND _poca_gemmi_hints "${GEMMI_ROOT}" "${GEMMI_ROOT}/include")
endif()
if(DEFINED ENV{GEMMI_ROOT})
    list(APPEND _poca_gemmi_hints "$ENV{GEMMI_ROOT}" "$ENV{GEMMI_ROOT}/include")
endif()
find_path(GEMMI_INCLUDE_DIR
    NAMES gemmi/mmread.hpp
    HINTS ${_poca_gemmi_hints})
unset(_poca_gemmi_hints)
if(NOT GEMMI_INCLUDE_DIR)
    message(FATAL_ERROR
        "Gemmi headers were not found. Set GEMMI_ROOT to the Gemmi repository root or its include directory.")
endif()
add_library(PoCA_gemmi INTERFACE)
target_include_directories(PoCA_gemmi INTERFACE "${GEMMI_INCLUDE_DIR}")
add_library(PoCA::Gemmi ALIAS PoCA_gemmi)

add_library(PoCA_openmp INTERFACE)
if(OpenMP_CXX_FOUND)
    target_link_libraries(PoCA_openmp INTERFACE OpenMP::OpenMP_CXX)
endif()
add_library(PoCA::OpenMP ALIAS PoCA_openmp)

# Normalize GLM to one target even when using PoCA's header-only FindGLM module.
if(NOT TARGET PoCA_glm)
    add_library(PoCA_glm INTERFACE)
    if(TARGET glm::glm)
        target_link_libraries(PoCA_glm INTERFACE glm::glm)
    elseif(TARGET GLM::GLM)
        target_link_libraries(PoCA_glm INTERFACE GLM::GLM)
    else()
        target_include_directories(PoCA_glm INTERFACE "${GLM_INCLUDE_DIRS}")
    endif()
    add_library(PoCA::GLM ALIAS PoCA_glm)
endif()

# Normalize OpenGL/GLEW to one interface target.
#
# On the legacy Windows/CMake 3.25 setup, the generic GLEW::GLEW imported
# target can resolve to the runtime glew32.dll, which MSVC cannot link
# directly.  The classic GLEW library list resolves to .lib files and is
# therefore preferred on Windows.  Do not blindly define GLEW_STATIC: some
# FindGLEW versions/install layouts report glew32.lib through
# GLEW_STATIC_LIBRARIES even though it is the DLL import library.  Defining
# GLEW_STATIC while linking that import library changes the expected symbols
# (__glew* instead of __imp___glew*) and produces LNK2019 errors.
add_library(PoCA_opengl_deps INTERFACE)
target_link_libraries(PoCA_opengl_deps INTERFACE OpenGL::GL)
if(WIN32 AND GLEW_STATIC_LIBRARIES)
    target_include_directories(PoCA_opengl_deps INTERFACE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(PoCA_opengl_deps INTERFACE ${GLEW_STATIC_LIBRARIES})

    # Standard Windows static GLEW libraries are named glew32s.lib /
    # glew32sd.lib.  Only those require GLEW_STATIC.  glew32.lib /
    # glew32d.lib are DLL import libraries and must be used without it.
    set(_poca_glew_true_static FALSE)
    foreach(_poca_glew_item IN LISTS GLEW_STATIC_LIBRARIES)
        if(_poca_glew_item STREQUAL "optimized" OR
           _poca_glew_item STREQUAL "debug" OR
           _poca_glew_item STREQUAL "general")
            continue()
        endif()
        get_filename_component(_poca_glew_name "${_poca_glew_item}" NAME_WE)
        string(TOLOWER "${_poca_glew_name}" _poca_glew_name_lower)
        if(_poca_glew_name_lower MATCHES "^glew32s(d)?$")
            set(_poca_glew_true_static TRUE)
            break()
        endif()
    endforeach()

    if(_poca_glew_true_static)
        target_compile_definitions(PoCA_opengl_deps INTERFACE GLEW_STATIC)
        message(STATUS "PoCA GLEW: static (${GLEW_STATIC_LIBRARIES})")
    else()
        message(STATUS "PoCA GLEW: DLL import libraries (${GLEW_STATIC_LIBRARIES})")
    endif()

    unset(_poca_glew_true_static)
    unset(_poca_glew_item)
    unset(_poca_glew_name)
    unset(_poca_glew_name_lower)
elseif(TARGET GLEW::GLEW)
    target_link_libraries(PoCA_opengl_deps INTERFACE GLEW::GLEW)
    message(STATUS "PoCA GLEW: imported target GLEW::GLEW")
else()
    target_include_directories(PoCA_opengl_deps INTERFACE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(PoCA_opengl_deps INTERFACE ${GLEW_LIBRARIES})
    message(STATUS "PoCA GLEW: ${GLEW_LIBRARIES}")
endif()
if(TARGET OpenGL::GLU)
    target_link_libraries(PoCA_opengl_deps INTERFACE OpenGL::GLU)
endif()
add_library(PoCA::OpenGLDeps ALIAS PoCA_opengl_deps)

# Normalize CGAL and its optional Core/Eigen support components. Modern CGAL 5
# and 6 packages expose CGAL::CGAL; retain a variable-based fallback for older
# local packages so the legacy machine does not need to change its install.
add_library(PoCA_cgal INTERFACE)
if(TARGET CGAL::CGAL)
    target_link_libraries(PoCA_cgal INTERFACE CGAL::CGAL)
else()
    if(CGAL_INCLUDE_DIRS)
        target_include_directories(PoCA_cgal INTERFACE ${CGAL_INCLUDE_DIRS})
    endif()
    target_link_libraries(PoCA_cgal INTERFACE ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})
endif()
if(TARGET CGAL::CGAL_Core)
    target_link_libraries(PoCA_cgal INTERFACE CGAL::CGAL_Core)
elseif(CGAL_Core_LIBRARIES)
    target_link_libraries(PoCA_cgal INTERFACE ${CGAL_Core_LIBRARIES})
endif()
target_link_libraries(PoCA_cgal INTERFACE TBB::tbb TBB::tbbmalloc)
if(TARGET CGAL::Eigen3_support)
    target_link_libraries(PoCA_cgal INTERFACE CGAL::Eigen3_support)
else()
    target_link_libraries(PoCA_cgal INTERFACE Eigen3::Eigen)
    target_compile_definitions(PoCA_cgal INTERFACE CGAL_EIGEN3_ENABLED)
endif()
target_compile_definitions(PoCA_cgal INTERFACE CGAL_LINKED_WITH_TBB)
add_library(PoCA::CGAL ALIAS PoCA_cgal)

# TBB 2020.x is supported by PoCA's custom FindTBB module, which provides namespaced targets.
add_library(PoCA_tbb INTERFACE)
target_link_libraries(PoCA_tbb INTERFACE TBB::tbb TBB::tbbmalloc)
add_library(PoCA::TBB ALIAS PoCA_tbb)

# Public headers shared by the main executable and dynamically loaded plugins.
add_library(PoCA_plugin_api INTERFACE)
target_include_directories(PoCA_plugin_api INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(PoCA_plugin_api INTERFACE Qt5::Core)
add_library(PoCA::PluginAPI ALIAS PoCA_plugin_api)
