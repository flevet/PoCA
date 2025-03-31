/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectColocalization.hpp
*
* Copyright: Florian Levet (2020-2021)
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

#ifndef ObjectColocalization_h__
#define ObjectColocalization_h__

//#include <Interfaces/PolygonInterface.hpp>
#include <General/BasicComponent.hpp>
#include <Geometry/ObjectLists.hpp>
#include <Geometry/ObjectListDelaunay.hpp>
#include <Geometry/CGAL_includes.hpp>


namespace poca::geometry {
	class MyObjectInterface;
	class DelaunayTriangulationInterface;
}

class ObjectColocalization : public poca::core::BasicComponent {
public:
	ObjectColocalization(poca::core::MyObjectInterface*, poca::core::MyObjectInterface*, const bool = true, const float = 0.5f, const uint32_t = 2, const uint32_t = 5);
	~ObjectColocalization();

	poca::core::BasicComponentInterface* copy();

	poca::core::BoundingBox computeBoundingBoxElement(const int) const;

	poca::geometry::DelaunayTriangulationInterface* getDelaunay() const { return m_delaunay; }
	poca::geometry::ObjectListInterface* getObjectsOriginalColor(const uint32_t _idx) const { return m_objectsPerColor[_idx]->currentObjectList(); }
	poca::geometry::ObjectListInterface* getObjectsOverlap() const { return m_objectList; }
	const std::vector <uint32_t>& getSelectionOverlap() const { return m_selectionIntersection; }
	const std::vector <float>& getXs() const { return m_xs; }
	const std::vector <float>& getYs() const { return m_ys; }
	const std::vector <float>& getZs() const { return m_zs; }
	const std::vector <poca::core::Vec3mf>& getTriangles() const { return m_triangles; }
	const uint32_t dimension() const { return m_dimension; }
	const poca::core::MyArrayUInt32& getInfosColor(const uint32_t _idx) const { return _idx == 0 ? getInfosColor1() : getInfosColor2(); }
	const poca::core::MyArrayUInt32& getInfosColor1() const { return m_infoColor1; }
	const poca::core::MyArrayUInt32& getInfosColor2() const { return m_infoColor2; }
	const std::vector <std::pair<uint32_t, uint32_t>>& getInfosColoc() const { return m_infoColoc; }
	const poca::core::MyArrayVec3mf& getLocsOverlapObjects() const { return m_origPointsInOverlapObjects; }

	const float getAreaObjectColor(const uint32_t, const uint32_t) const;
	const float getAreaObjectIntersection(const uint32_t) const;

	const std::vector <bool>& getSelectedTetra() const {
		return m_selectedTetrahedra;
	}

protected:
	void computeOverlap();
	void computeOverlap2();

	void computeOverlapLast();
	void computeOverlapLast2();

protected:
	poca::core::MyObjectInterface* m_objects[2];
	poca::geometry::ObjectLists* m_objectsPerColor[2];

	bool m_samplingEnabled;
	float m_distance2D;
	uint32_t m_subdiv3D, m_minNbLocs;

	std::vector <float> m_xs, m_ys, m_zs;
	std::vector <float> m_xsOrigPoints, m_ysOrigPoints, m_zsOrigPoints;
	poca::geometry::DelaunayTriangulationInterface* m_delaunay;
	poca::geometry::ObjectListDelaunay* m_objectList;
	std::vector <uint32_t> m_selectionIntersection;

	poca::core::MyArrayUInt32 m_infoColor1, m_infoColor2;
	std::vector <std::pair<uint32_t, uint32_t>> m_infoColoc;

	std::vector <poca::core::Vec3mf> m_triangles;

	size_t m_nbObjectsPerColor[2];
	std::vector <uint32_t> m_linkLocsToObjectsBothColor;

	std::vector <std::array<uint32_t, 2>> m_linkLocsToObjectsBothColor2;

	uint32_t m_dimension;

	std::vector <bool> m_selectedTetrahedra;
	poca::core::MyArrayVec3mf m_origPointsInOverlapObjects;
	//poca::geometry::PolygonInterfaceList m_polygons[2];
};
#endif // ObjectColocalization_h__

