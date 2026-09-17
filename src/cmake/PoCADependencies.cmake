# Central third-party dependency discovery for PoCA.
# Dependencies are found once here; individual targets only state what they link to.

# Optional root hints exposed in CMake GUI. Leave them empty when the package is already discoverable.
set(GLEW_ROOT "" CACHE PATH "Root directory of the GLEW installation")
set(GLM_ROOT "" CACHE PATH "Root directory containing glm/glm.hpp")
set(TBB_ROOT "" CACHE PATH "Root directory of the Intel TBB installation")
set(Boost_ROOT "" CACHE PATH "Root directory of the Boost headers used by CGAL")
set(Eigen3_ROOT "" CACHE PATH "Eigen3 source/install root containing Eigen/Core, or an installation prefix")
set(GEMMI_ROOT "" CACHE PATH "Gemmi repository root or include directory")

find_package(Qt5 5.15 REQUIRED COMPONENTS Core Gui Widgets OpenGL PrintSupport)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLM REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb tbbmalloc)

# Eigen must be discoverable before CGAL is configured. Several CGAL algorithms
# (Poisson reconstruction and mean-curvature-flow skeletonization) select their
# default linear algebra solver from Eigen at configuration/include time.
find_package(Eigen3 3.1 REQUIRED)

# Some older Eigen/CGAL FindEigen3 modules report EIGEN3_INCLUDE_DIR but do
# not provide the modern Eigen3::Eigen imported target.  Normalize that here
# so every PoCA target can use the same target-based dependency interface.
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

# CGAL 6.0.x still uses CMake's legacy FindBoost module internally.
# CMake 3.30+ policy CMP0167 disables that module in NEW mode, so keep
# OLD behavior only while configuring this older CGAL release.
if(POLICY CMP0167)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0167 OLD)
endif()
find_package(CGAL 6.0 REQUIRED COMPONENTS Core)
if(POLICY CMP0167)
    cmake_policy(POP)
endif()

find_package(TinyTIFF REQUIRED CONFIG)
find_package(tinysplinecxx REQUIRED CONFIG)
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
add_library(PoCA_opengl_deps INTERFACE)
target_link_libraries(PoCA_opengl_deps INTERFACE OpenGL::GL GLEW::GLEW)
if(TARGET OpenGL::GLU)
    target_link_libraries(PoCA_opengl_deps INTERFACE OpenGL::GLU)
endif()
add_library(PoCA::OpenGLDeps ALIAS PoCA_opengl_deps)

# Normalize CGAL and its optional Core/Eigen support components. PoCA uses
# Eigen-dependent CGAL algorithms (notably Poisson reconstruction and
# mean-curvature-flow skeletonization), so Eigen support is mandatory here.
add_library(PoCA_cgal INTERFACE)
target_link_libraries(PoCA_cgal INTERFACE CGAL::CGAL TBB::tbb TBB::tbbmalloc)
if(TARGET CGAL::CGAL_Core)
    target_link_libraries(PoCA_cgal INTERFACE CGAL::CGAL_Core)
endif()
if(TARGET CGAL::Eigen3_support)
    target_link_libraries(PoCA_cgal INTERFACE CGAL::Eigen3_support)
else()
    # Fallback for CGAL packages that do not expose the support target.
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
