#ifndef DelaunayTriangulationFactory_h__
#define DelaunayTriangulationFactory_h__

#include <Geometry/DelaunayTriangulation.hpp>

namespace poca::geometry {
	class DelaunayTriangulationFactory {
	public:
		DelaunayTriangulationFactory();
		~DelaunayTriangulationFactory();

		DelaunayTriangulation* createDelaunayTriangulation(const std::vector <float>&, const std::vector <float>&);
		DelaunayTriangulation* createDelaunayTriangulation(const std::vector <float>&, const std::vector <float>&, const std::vector <float>&);
	};
}

#endif // DelaunayTriangulationFactory_h__