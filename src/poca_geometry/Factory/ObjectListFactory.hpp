/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListFactory.hpp
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

#ifndef ObjectListFactory_h__
#define ObjectListFactory_h__

#include <map>
#include <set>
#include <vector>

#include <General/Vec3.hpp>
#include <Geometry/CGAL_includes.hpp>

#include "../Interfaces/ObjectListFactoryInterface.hpp"

namespace poca::geometry {
	class ObjectListFactory: public ObjectListFactoryInterface {
	public:
		ObjectListFactory();
		~ObjectListFactory();

		ObjectListInterface* createObjectList(poca::core::MyObjectInterface*, const std::vector <bool>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max(), const bool = false);
		ObjectListInterface* createObjectListFromDelaunay(poca::core::MyObjectInterface*, const std::vector <bool>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max(), const bool = false);
		ObjectListInterface* createObjectListAlreadyIdentified(poca::core::MyObjectInterface*, const std::vector <uint32_t>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max());

		ObjectListInterface* createObjectList(DelaunayTriangulationInterface*, const std::vector <bool>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max(), const std::vector <poca::core::ROIInterface*>& = std::vector <poca::core::ROIInterface*>());
		ObjectListInterface* createObjectListFromDelaunay(DelaunayTriangulationInterface*, const std::vector <bool>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max(), const std::vector <poca::core::ROIInterface*>& = std::vector <poca::core::ROIInterface*>());
		ObjectListInterface* createObjectListAlreadyIdentified(DelaunayTriangulationInterface*, const std::vector <uint32_t>&, const float = std::numeric_limits < float >::max(), const size_t = 3, const size_t = std::numeric_limits < float >::max(), const float = 0.f, const float = std::numeric_limits < float >::max());

		ObjectListInterface* createObjectList2D(DelaunayTriangulationInterface*, const std::vector <bool>&, const float, const size_t, const size_t, const float, const float, const std::vector <poca::core::ROIInterface*>& = std::vector <poca::core::ROIInterface*>());
		ObjectListInterface* createObjectList3D(DelaunayTriangulationInterface*, const std::vector <bool>&, const float, const size_t, const size_t, const float, const float, const std::vector <poca::core::ROIInterface*> & = std::vector <poca::core::ROIInterface*>());

		//ObjectList* createObjectList2D(DelaunayTriangulationInterface*, const std::vector <uint32_t>&, const float, const size_t, const size_t, const float, const float, const std::vector <poca::core::ROIInterface*> & = std::vector <poca::core::ROIInterface*>());
		//ObjectList* createObjectList3D(DelaunayTriangulationInterface*, const std::vector <uint32_t>&, const float, const size_t, const size_t, const float, const float, const std::vector <poca::core::ROIInterface*> & = std::vector <poca::core::ROIInterface*>());

		ObjectListInterface* createObjectList2D(DelaunayTriangulationInterface*, const std::map <uint32_t, std::vector <uint32_t>>&, const float, const size_t, const size_t, const float, const float);
		ObjectListInterface* createObjectList3D(DelaunayTriangulationInterface*, const std::map <uint32_t, std::vector <uint32_t>>&, const float, const size_t, const size_t, const float, const float);

	protected:
		void computeConvexHullObject2D(const float*, const float*, const float*, const std::vector <uint32_t>&, std::vector <poca::core::Vec3mf>&, std::vector <poca::core::Vec3mf>&, float&);
		void computeConvexHullObject3D(const float*, const float*, const float*, const std::vector <uint32_t>&, std::vector <poca::core::Vec3mf>&, std::vector <poca::core::Vec3mf>&, float&);
		void computeConvexHullObject2DFromOutline(const float *, const float*, const float*, const std::set <uint32_t>&, std::vector <poca::core::Vec3mf>&, float&);
		void computeConvexHullObject3DFromOutline(const float*, const float*, const float*, const std::set <uint32_t>&, std::vector <poca::core::Vec3mf>&, float&);
		
		void computePoissonSurfaceObject(const float*, const float*, const float*, const std::set <uint32_t>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&, std::vector <poca::core::Vec3mf>&, float&);
		void computeAlphaShape(const float*, const float*, const float*, const std::set <uint32_t>&, std::vector <poca::core::Vec3mf>&, float&);
	
		void computeNormalOfLocsObject(const std::set <uint32_t>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&, std::vector <poca::core::Vec3mf>&);
	};
}

#endif // ObjectListFactory_h__

