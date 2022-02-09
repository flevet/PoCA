/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulationInterface.hpp
*
* Copyright: Florian Levet (2020-2022)
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

#ifndef DelaunayTriangulationInterface_h__
#define DelaunayTriangulationInterface_h__

#include <vector>
#include <array>

#include <General/BasicComponent.hpp>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>

namespace poca::geometry {
	class DelaunayTriangulationInterface : public poca::core::BasicComponent {
	public:
		virtual ~DelaunayTriangulationInterface() = default;

		virtual poca::core::BasicComponent* copy() = 0;

		virtual void generateTriangles(std::vector <poca::core::Vec3mf>&) = 0;
		virtual void getFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const = 0;
		virtual void generatePickingIndices(std::vector <float>&) const = 0;
		virtual poca::core::BoundingBox computeBoundingBoxElement(const int) const = 0;
		virtual poca::core::Vec3mf computeBarycenterElement(const int) const = 0;
		virtual void getVerticesTriangle(const size_t, std::vector <poca::core::Vec3mf>&) const = 0;
		virtual const uint32_t dimension() const = 0;
		virtual void generateFaceSelectionFromLocSelection(const std::vector <bool>&, std::vector <bool>&) = 0;
		virtual void generateFaceSelectionFromLocSelection(const std::vector <uint32_t>&, std::vector <uint32_t>&) = 0;
		virtual void generateFaceSelectionFromLocSelection(const std::vector <uint32_t>&, std::map <uint32_t, std::vector<uint32_t>>&) = 0;
		virtual void getTrianglesNeighboringPoint(uint32_t, std::vector <uint32_t>&) = 0;
		virtual bool isPointInTriangle(const float, const float, const float, const uint32_t) const = 0;
		virtual void trianglesAdjacentToTriangle(std::uint32_t, std::vector <std::uint32_t>&) = 0;
		virtual uint32_t indexTriangleOfPoint(const float, const float, const float, const uint32_t = std::numeric_limits<std::uint32_t>::max()) = 0;
		virtual const std::array<size_t, 3> getOutline(const size_t, const size_t) const = 0;

		virtual const std::vector<float>& xs() const { return m_xs; }
		virtual const std::vector<float>& ys() const { return m_ys; }
		virtual const std::vector<float>& zs() const { return m_zs; }
		virtual const float* getXs() const { return m_xs.data(); }
		virtual const float* getYs() const { return m_ys.data(); }
		virtual const float* getZs() const { return !m_zs.empty() ? m_zs.data() : NULL; }
		virtual const size_t nbPoints() const { return m_xs.size(); }
		virtual const poca::core::MyArrayUInt32& getNeighbors() const { return m_neighbors; }
		virtual const size_t nbFaces() const { return m_nbCells; }
		virtual const std::vector<uint32_t>& getTriangles() const { return m_triangles; }
	protected:
		DelaunayTriangulationInterface(const std::vector<float>& _xs, const std::vector<float>& _ys, const std::vector<size_t>& _triangles, const size_t _nbCells) :BasicComponent("DelaunayTriangulation"), m_xs(_xs), m_ys(_ys), m_zs(NULL), m_triangles(_triangles.begin(), _triangles.end()), m_nbCells(_nbCells) {}
		DelaunayTriangulationInterface(const std::vector<float>& _xs, const std::vector<float>& _ys, const std::vector<float>& _zs, const std::vector<size_t>& _triangles, const size_t _nbCells) :BasicComponent("DelaunayTriangulation"), m_xs(_xs), m_ys(_ys), m_zs(_zs), m_triangles(_triangles.begin(), _triangles.end()), m_nbCells(_nbCells) {}
		DelaunayTriangulationInterface(const std::vector<float>& _xs, const std::vector<float>& _ys, const std::vector<uint32_t>& _triangles, const size_t _nbCells) :BasicComponent("DelaunayTriangulation"), m_xs(_xs), m_ys(_ys), m_zs(NULL), m_triangles(_triangles), m_nbCells(_nbCells) {}
		DelaunayTriangulationInterface(const std::vector<float>& _xs, const std::vector<float>& _ys, const std::vector<float>& _zs, const std::vector<uint32_t>& _triangles, const size_t _nbCells) :BasicComponent("DelaunayTriangulation"), m_xs(_xs), m_ys(_ys), m_zs(_zs), m_triangles(_triangles), m_nbCells(_nbCells) {}

	protected:
		poca::core::MyArrayUInt32 m_neighbors;
		std::vector<uint32_t> m_triangles;
		size_t m_nbCells;
		std::vector <float>  m_xs, m_ys, m_zs;
	};
}

#endif

