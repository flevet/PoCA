/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      GeometryCommandContext.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef GeometryCommandContext_h__
#define GeometryCommandContext_h__

#include <vector>

#include <General/Command.hpp>
#include <General/Vec3.hpp>

namespace poca::geometry {
	struct DetectionSetNormalsContext {
		const std::vector<poca::core::Vec3mf>* normals = nullptr;
	};

	class DetectionSet;
	class ObjectListMesh;
	class ObjectListInterface;

	struct CleanedDetectionSetContext {
		poca::geometry::DetectionSet* dset = nullptr;
	};

	struct CreatedObjectListMeshContext {
		poca::geometry::ObjectListMesh* objects = nullptr;
	};

	struct CreatedObjectListContext {
		poca::geometry::ObjectListInterface* objects = nullptr;
	};
}

#endif
