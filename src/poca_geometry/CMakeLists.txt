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
  Geometry/DelaunayTriangulation.hpp
  Geometry/DetectionSet.hpp
  Geometry/GeometryCommandContext.hpp
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
  Geometry/DelaunayTriangulation.cpp
  Geometry/DetectionSet.cpp
  Geometry/ObjectLists.cpp
  Geometry/ObjectListDelaunay.cpp
  Geometry/ObjectListMesh.cpp
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

poca_add_static_library(${targetname}
    ALIAS PoCA::Geometry
    SOURCES ${header_list} ${source_list}
    USE_CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS
    CUDA_SOURCES ${cuda_list}
    PUBLIC_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}
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
