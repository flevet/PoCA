set(targetname poca_geometry)

set(3D_VORONOI_GPU_HEADER_FILES
  3D_voronoi_GPU/voronoi.h
)

set(INTERFACES_HEADER_FILES
  Interfaces/ObjectFeaturesFactoryInterface.hpp
  Interfaces/ObjectListFactoryInterface.hpp
  Interfaces/ObjectListInterface.hpp
  Interfaces/ObjectIndicesFactoryInterface.hpp
  Interfaces/VoronoiDiagramFactoryInterface.hpp
  Interfaces/DelaunayTriangulationFactoryInterface.hpp
  Interfaces/DelaunayTriangulationInterface.hpp
  Interfaces/PolygonFactoryInterface.hpp
  Interfaces/PolygonInterface.hpp
)

set(FACTORY_HEADER_FILES
  Factory/ObjectListFactory.hpp
  Factory/ObjectFeaturesFactory.hpp
  Factory/VoronoiDiagramFactory.hpp
  Factory/DelaunayTriangulationFactory.hpp
  Factory/PolygonFactory.hpp
  Factory/ObjectIndicesFactory.hpp
)

set(FACTORY_SOURCE_FILES
  Factory/ObjectListFactory.cpp
  Factory/ObjectFeaturesFactory.cpp
  Factory/VoronoiDiagramFactory.cpp
  Factory/DelaunayTriangulationFactory.cpp
  Factory/PolygonFactory.cpp
  Factory/ObjectIndicesFactory.cpp
)

set(GEOMETRY_HEADER_FILES
  Geometry/BasicComputation.hpp
  Geometry/PCA.hpp
  Geometry/UMAP.hpp
  Geometry/DelaunayTriangulation.hpp
  Geometry/DetectionSet.hpp
  Geometry/GeometryCommandContext.hpp
  Geometry/MeshRepair.hpp
  Geometry/nanoflann.hpp
  Geometry/ObjectLists.hpp
  Geometry/ObjectListDelaunay.hpp
  Geometry/ObjectListMesh.hpp
  Geometry/ObjectListPolygon.hpp
  Geometry/StraightLine.hpp
  Geometry/Skeleton.hpp
  Geometry/VoronoiDiagram.hpp
  Geometry/Voronoi3DCellFeatures.hpp
  Geometry/CGAL_includes.hpp
  Geometry/CGAL_helpers.hpp
  Geometry/delaunator.hpp
  Geometry/voronator.hpp
  Geometry/dbscan.h
  Geometry/kdtree.h
  Geometry/Polygon.hpp
  Geometry/DelaunayOnSphere.hpp
  Geometry/TrackSet.hpp
)

set(GEOMETRY_SOURCE_FILES
  Geometry/BasicComputation.cpp
  Geometry/PCA.cpp
  Geometry/UMAP.cpp
  Geometry/DelaunayTriangulation.cpp
  Geometry/DetectionSet.cpp
  Geometry/DetectionSetStorage.cpp
  Geometry/MeshRepair.cpp
  Geometry/ObjectLists.cpp
  Geometry/ObjectListDelaunay.cpp
  Geometry/ObjectListMesh.cpp
  Geometry/ObjectListMeshPersistence.cpp
  Geometry/ObjectListMeshIndexedGeometry.cpp
  Geometry/ObjectListPolygon.cpp
  Geometry/StraightLine.cpp
  Geometry/Skeleton.cpp
  Geometry/VoronoiDiagram.cpp
  Geometry/Voronoi3DCellFeatures.cpp
  Geometry/CGAL_helpers.cpp
  Geometry/delaunator.cpp
  Geometry/voronator.cpp
  Geometry/kdtree.cpp
  Geometry/Polygon.cpp
  Geometry/DelaunayOnSphere.cpp
  Geometry/TrackSet.cpp
)

set(header_list
  ${3D_VORONOI_GPU_HEADER_FILES}
  ${INTERFACES_HEADER_FILES}
  ${FACTORY_HEADER_FILES}
  ${GEOMETRY_HEADER_FILES}
)

set(source_list
  ${FACTORY_SOURCE_FILES}
  ${GEOMETRY_SOURCE_FILES}
)

set(cuda_list
  3D_voronoi_GPU/voronoi.cu
)

# UMAP++ and its header-only dependency chain are current-main functionality.
# Populate pinned releases without executing their CMake projects, avoiding a
# second Eigen discovery/configuration and keeping both PoCA toolchains on the
# same dependency versions.
include(FetchContent)
macro(poca_populate_header_dependency dependency repository tag)
  FetchContent_Declare(${dependency}
    GIT_REPOSITORY ${repository}
    GIT_TAG ${tag}
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR __poca_header_only_no_cmake__
  )
  FetchContent_MakeAvailable(${dependency})
endmacro()

poca_populate_header_dependency(poca_umappp https://github.com/libscran/umappp.git v3.3.2)
poca_populate_header_dependency(poca_aarand https://github.com/LTLA/aarand.git v1.1.0)
poca_populate_header_dependency(poca_subpar https://github.com/LTLA/subpar.git v0.5.0)
poca_populate_header_dependency(poca_sanisizer https://github.com/LTLA/sanisizer.git v0.2.0)
poca_populate_header_dependency(poca_knncolle https://github.com/knncolle/knncolle.git v3.1.0)
poca_populate_header_dependency(poca_irlba https://github.com/libscran/irlba.git v3.1.0)

set(_poca_umap_include_dirs
  ${poca_umappp_SOURCE_DIR}/include
  ${poca_aarand_SOURCE_DIR}/include
  ${poca_subpar_SOURCE_DIR}/include
  ${poca_sanisizer_SOURCE_DIR}/include
  ${poca_knncolle_SOURCE_DIR}/include
  ${poca_irlba_SOURCE_DIR}/include
)

poca_add_static_library(${targetname}
    ALIAS PoCA::Geometry
    SOURCES ${header_list} ${source_list}
    USE_CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS
    CUDA_SOURCES ${cuda_list}
    PUBLIC_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE_INCLUDE_DIRECTORIES
        ${_poca_umap_include_dirs}
    PUBLIC_LINK_LIBRARIES
        PoCA::Core
        Qt5::Core
        PoCA::CGAL
        PoCA::OpenGLDeps
        tinysplinecxx::tinysplinecxx
    PRIVATE_LINK_LIBRARIES
        Eigen3::Eigen
        PoCA::TBB
)
