/*
* Software:  PoCA: Point Cloud Analyst
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*/

#pragma once

#include <string>
#include <vector>

#include <Geometry/CGAL_includes.hpp>

namespace poca::objectlist {
	struct MeshSelfIntersectionRepairResult {
		std::vector<Surface_mesh_3_double> meshes;
		std::vector<float> sourceMeshIndices;
		std::string report;
	};

	MeshSelfIntersectionRepairResult repairSelfIntersections(const std::vector<Surface_mesh_3_double>&);
}
