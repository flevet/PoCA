/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef DatasetPlacement_hpp__
#define DatasetPlacement_hpp__
#include <General/Vec6.hpp>
#include <string>
#include <vector>
namespace poca::core {
	struct PlacementCircle {
		float x, y, radius;
		std::string label;
	};
	std::vector<BoundingBox> computeRegularGridPlacement(const std::vector<BoundingBox>&, int, int);
	std::vector<BoundingBox> computePackPlacement(const std::vector<BoundingBox>&, double);
	std::vector<BoundingBox> computeCirclePlacement(const std::vector<BoundingBox>&,
		const std::vector<std::vector<size_t>>&, const std::vector<std::string>&, std::vector<PlacementCircle>* = nullptr);
}
#endif
