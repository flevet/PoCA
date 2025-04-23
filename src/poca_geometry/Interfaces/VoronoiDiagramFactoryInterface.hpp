/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramFactoryInterface.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*
* PoCA is a free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 3 of the License, or (at your option) any later version.
*
* The algorithms that underlie PoCA have required considerable
* development. They are described in the original SR-Tesseler paper,
* doi:10.1038/nmeth.3579. If you use PoCA as part of work (visualization, 
* manipulation, quantification) towards a scientific publication, please include 
* a citation to the original paper.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program; if not, write to the Free Software Foundation,
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#ifndef VoronoiDiagramFactoryInterface_h__
#define VoronoiDiagramFactoryInterface_h__

#include <vector>

#include <General/Vec6.hpp>

#include "../Geometry/DetectionSet.hpp"

namespace poca::core {
	class MyObjectInterface;
	class PluginList;
}

namespace poca::geometry {
	class VoronoiDiagram;
	class DelaunayTriangulationInterface;

	class VoronoiDiagramFactoryInterface {
	public:
		virtual VoronoiDiagram* createVoronoiDiagram(poca::core::MyObjectInterface*, bool, poca::core::PluginList*, const bool = true) = 0;
		virtual VoronoiDiagram* createVoronoiDiagramOnSphere(poca::core::MyObjectInterface*, bool, poca::core::PluginList*, const bool = true) = 0;
		
		virtual VoronoiDiagram* createVoronoiDiagram(const std::vector <float>&, const std::vector <float>&, const poca::core::BoundingBox&, KdTree_DetectionPoint* = NULL, DelaunayTriangulationInterface* = NULL) = 0;
		virtual VoronoiDiagram* createVoronoiDiagram(const std::vector <float>&, const std::vector <float>&, const std::vector <float>&, KdTree_DetectionPoint* = NULL, DelaunayTriangulationInterface* = NULL, const bool = true) = 0;
		virtual VoronoiDiagram* createVoronoiDiagramOnSphere(const std::vector <float>&, const std::vector <float>&, const std::vector <float>&, KdTree_DetectionPoint* = NULL, DelaunayTriangulationInterface* = NULL, const bool = true) = 0;
	};

	VoronoiDiagramFactoryInterface* createVoronoiDiagramFactory();
}

#endif

