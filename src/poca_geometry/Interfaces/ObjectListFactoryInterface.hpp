/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListFactoryInterface.hpp
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

#ifndef ObjectListFactoryInterface_h__
#define ObjectListFactoryInterface_h__

#include <vector>

namespace poca::core {
	class MyObjectInterface;
	class ROIInterface;
}

namespace poca::geometry {
	class DelaunayTriangulationInterface;
	class ObjectListInterface;

	class ObjectListFactoryInterface {
	public:
		enum TypeShape { TRIANGULATION = 0, CONVEX_HULL = 1, POISSON_SURFACE = 2, ALPHA_SHAPE = 3 };
		
		virtual ObjectListInterface* createObjectList(poca::core::MyObjectInterface*, const std::vector <bool>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max(), const bool = false) = 0;
		virtual ObjectListInterface* createObjectListFromDelaunay(poca::core::MyObjectInterface*, const std::vector <bool>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max(), const bool = false) = 0;
		virtual ObjectListInterface* createObjectListAlreadyIdentified(poca::core::MyObjectInterface*, const std::vector <uint32_t>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max()) = 0;

		virtual ObjectListInterface* createObjectList(DelaunayTriangulationInterface*, const std::vector <bool>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max(), const std::vector <poca::core::ROIInterface*>& = std::vector <poca::core::ROIInterface*>()) = 0;
		virtual ObjectListInterface* createObjectListFromDelaunay(DelaunayTriangulationInterface*, const std::vector <bool>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max(), const std::vector <poca::core::ROIInterface*>& = std::vector <poca::core::ROIInterface*>()) = 0;
		virtual ObjectListInterface* createObjectListAlreadyIdentified(DelaunayTriangulationInterface*, const std::vector <uint32_t>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max()) = 0;
	
		static TypeShape getTypeId(const std::string& _typeS)
		{
			if (_typeS == "triangulation")
				return TRIANGULATION;
			if (_typeS == "convex_hull")
				return CONVEX_HULL;
			if (_typeS == "poisson_surface")
				return POISSON_SURFACE;
			if (_typeS == "alpha_shape")
				return ALPHA_SHAPE;
		}

		static std::string getTypeStr(const int _id)
		{
			std::string type;
			switch (_id) {
			case TRIANGULATION:
				type = "triangulation";
				break;
			case CONVEX_HULL:
				type = "convex_hull";
				break;
			case POISSON_SURFACE:
				type = "poisson_surface";
				break;
			case ALPHA_SHAPE:
				type = "alpha_shape";
				break;
			}
			return type;
		}

	protected:
	};

	ObjectListFactoryInterface* createObjectListFactory();
}

#endif // DelaunayTriangulationFactory_h__

