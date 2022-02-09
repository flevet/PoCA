#ifndef VoronoiDiagramFactory_h__
#define VoronoiDiagramFactory_h__

#include <vector>

#include <General/Vec6.hpp>
#include <Geometry/VoronoiDiagram.hpp>
#include <Geometry/DetectionSet.hpp>
#include <Interfaces/DelaunayTriangulationInterface.hpp>

namespace poca::geometry {
	class VoronoiDiagramFactory {
	public:
		VoronoiDiagramFactory();
		~VoronoiDiagramFactory();

		VoronoiDiagram* createVoronoiDiagram(const std::vector <float>&, const std::vector <float>&, const poca::core::BoundingBox &, KdTree_DetectionPoint * = NULL, DelaunayTriangulationInterface* = NULL);
		VoronoiDiagram* createVoronoiDiagram(const std::vector <float>&, const std::vector <float>&, const std::vector <float>&, KdTree_DetectionPoint * = NULL, DelaunayTriangulationInterface* = NULL, const bool = true);
	};
}

#endif // VoronoiDiagramFactory_h__