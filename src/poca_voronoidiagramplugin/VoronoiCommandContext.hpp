/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiCommandContext.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef VoronoiCommandContext_h__
#define VoronoiCommandContext_h__

#include <General/Command.hpp>

namespace poca::geometry {
	class DetectionSet;
}

namespace poca::voronoi {
	struct CreatedDetectionSetContext {
		poca::geometry::DetectionSet* dset = nullptr;
	};
}

#endif
