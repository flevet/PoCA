/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulation.cpp
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

#include <algorithm>
#include <math.h>

#include <General/MyData.hpp>
#include <General/Misc.h>

#include "DelaunayTriangulation.hpp"
#include "BasicComputation.hpp"
#include "delaunator.hpp"

namespace poca::geometry {

	DelaunayTriangulation2DOnSphere::DelaunayTriangulation2DOnSphere(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector <float>& _zs, 
		const std::vector<size_t>& _triangles, const poca::core::MyArrayUInt32& _neigbors, CGALDelaunayOnSphere* _delau, const poca::core::Vec3mf& _centroid, const float _radius) : DelaunayTriangulation2D(_xs, _ys, _zs, _triangles, _neigbors), m_internalDelaunay(_delau),
																																						m_centroid(_centroid), m_radius(_radius)
	{
	}

	DelaunayTriangulation2DOnSphere::~DelaunayTriangulation2DOnSphere()
	{
	}

	poca::core::BasicComponentInterface* DelaunayTriangulation2DOnSphere::copy()
	{
		return static_cast<poca::core::BasicComponentInterface*>(new DelaunayTriangulation2DOnSphere(*this));
	}

	void DelaunayTriangulation2DOnSphere::getTrianglesNeighboringPoint(uint32_t _indexPoint, std::vector <uint32_t>& _neighs)
	{
	}

	void DelaunayTriangulation2DOnSphere::trianglesAdjacentToTriangle(std::uint32_t _tri, std::vector <std::uint32_t>& _neighs)
	{
	}

	uint32_t DelaunayTriangulation2DOnSphere::indexTriangleOfPoint(const float _x, const float _y, const float _z, const uint32_t _idx)
	{
		return 0;
	}

	const std::array<size_t, 3> DelaunayTriangulation2DOnSphere::getOutline(const size_t _idxTriangle, const size_t _idxNeigh) const
	{
		return std::array<size_t, 3>();
	}

	DelaunayTriangulation2DDelaunator::DelaunayTriangulation2DDelaunator(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector<size_t>& _triangles, const poca::core::MyArrayUInt32& _neigbors, delaunator::DelaunayFromDelaunator* _delau, std::vector <double>* _coords) : DelaunayTriangulation2D(_xs, _ys, _triangles, _neigbors), m_internalDelaunay(_delau), m_coords(_coords)
	{
	}

	DelaunayTriangulation2DDelaunator::~DelaunayTriangulation2DDelaunator()
	{
		delete m_coords;
	}

	poca::core::BasicComponentInterface* DelaunayTriangulation2DDelaunator::copy()
	{
		return new DelaunayTriangulation2DDelaunator(*this);
	}
	
	void DelaunayTriangulation2DDelaunator::getTrianglesNeighboringPoint(uint32_t _indexPoint, std::vector <uint32_t>& _neighs)
	{
		_neighs.clear();
		m_internalDelaunay->trianglesAdjacentToPointUint32(_indexPoint, _neighs);
	}

	void DelaunayTriangulation2DDelaunator::trianglesAdjacentToTriangle(std::uint32_t _tri, std::vector <std::uint32_t>& _neighs)
	{
		m_internalDelaunay->trianglesAdjacentToTriangleUint32(_tri, _neighs);
	}

	uint32_t DelaunayTriangulation2DDelaunator::indexTriangleOfPoint(const float _x, const float _y, const float _z, const uint32_t _idx)
	{
		return m_internalDelaunay->findTriangle(_x, _y, _idx);
	}

	const std::array<size_t, 3> DelaunayTriangulation2DDelaunator::getOutline(const size_t _idxTriangle, const size_t _idxNeigh) const
	{
		return m_internalDelaunay->getEdge(_idxTriangle, _idxNeigh);
	}

	DelaunayTriangulation2D::DelaunayTriangulation2D(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector<size_t>& _triangles, const poca::core::MyArrayUInt32& _neigbors) : DelaunayTriangulationInterface(_xs, _ys, _triangles, _triangles.size() / 3)
	{
		m_neighbors = _neigbors;

		//Create area feature
		std::vector <float> areas(nbFaces());
		for (size_t n = 0; n < m_triangles.size(); n += 3) {
			size_t i1 = m_triangles[n], i2 = m_triangles[n + 1], i3 = m_triangles[n + 2];
			float sideA = poca::geometry::distance<float>(m_xs[i1], m_ys[i1], m_xs[i2], m_ys[i2]), sideB = poca::geometry::distance<float>(m_xs[i1], m_ys[i1], m_xs[i3], m_ys[i3]), sideC = poca::geometry::distance<float>(m_xs[i3], m_ys[i3], m_xs[i2], m_ys[i2]);
			areas[n / 3] = computeAreaTriangle<float>(sideA, sideB, sideC);
		}
		m_data["area"] = poca::core::generateDataWithLog(areas);
		m_selection.resize(areas.size());
		setCurrentHistogramType("area");
		forceRegenerateSelection();
	}

	DelaunayTriangulation2D::DelaunayTriangulation2D(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector <float>& _zs, const std::vector<size_t>& _triangles, const poca::core::MyArrayUInt32& _neigbors) : DelaunayTriangulationInterface(_xs, _ys, _zs, _triangles, _triangles.size() / 3)
	{
		m_neighbors = _neigbors;

		//Create area feature
		std::vector <float> areas(nbFaces());
		for (size_t n = 0; n < m_triangles.size(); n += 3) {
			size_t i1 = m_triangles[n], i2 = m_triangles[n + 1], i3 = m_triangles[n + 2];
			float sideA = poca::geometry::distance(m_xs[i1], m_ys[i1], m_zs[i1], m_xs[i2], m_ys[i2], m_zs[i2]),
				sideB = poca::geometry::distance(m_xs[i1], m_ys[i1], m_zs[i1], m_xs[i3], m_ys[i3], m_zs[i3]),
				sideC = poca::geometry::distance(m_xs[i3], m_ys[i3], m_zs[i3], m_xs[i2], m_ys[i2], m_zs[i2]);
			areas[n / 3] = computeAreaTriangle<float>(sideA, sideB, sideC);
		}
		m_data["area"] = poca::core::generateDataWithLog(areas);
		m_selection.resize(areas.size());
		setCurrentHistogramType("area");
		forceRegenerateSelection();
	}

	DelaunayTriangulation2D::~DelaunayTriangulation2D()
	{
	}

	void DelaunayTriangulation2D::generateTriangles(std::vector <poca::core::Vec3mf>& _triangles)
	{
		_triangles.resize(nbFaces() * 3);
		if (m_zs.empty()) {
			for (size_t n = 0; n < m_triangles.size(); n += 3) {
				size_t i1 = m_triangles[n], i2 = m_triangles[n + 1], i3 = m_triangles[n + 2];
				_triangles[n].set(m_xs[i1], m_ys[i1], 0.f);
				_triangles[n + 1].set(m_xs[i3], m_ys[i3], 0.f);
				_triangles[n + 2].set(m_xs[i2], m_ys[i2], 0.f);
			}
		}
		else {
			for (size_t n = 0; n < m_triangles.size(); n += 3) {
				size_t i1 = m_triangles[n], i2 = m_triangles[n + 1], i3 = m_triangles[n + 2];
				_triangles[n].set(m_xs[i1], m_ys[i1], m_zs[i1]);
				_triangles[n + 1].set(m_xs[i3], m_ys[i3], m_zs[i3]);
				_triangles[n + 2].set(m_xs[i2], m_ys[i2], m_zs[i2]);
			}
		}
	}

	void DelaunayTriangulation2D::getFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		_features.resize(m_triangles.size());

		uint32_t cpt = 0;
		for (unsigned int n = 0; n < nbFaces(); n++) {
			for (unsigned int i = 0; i < 3; i++)//3 vertices per triangle
				_features[cpt++] = _selection[n] ? _values[n] : _notSelectedValue;
		}
	}

	void DelaunayTriangulation2D::generatePickingIndices(std::vector <float>& _ids) const
	{
		_ids.resize(m_triangles.size());
		int curId = 0;
		std::generate(_ids.begin(), _ids.end(), [&, cpt = 0]() mutable { if (cpt % 3 == 0) curId++; cpt++; return curId; });
	}

	poca::core::BoundingBox DelaunayTriangulation2D::computeBoundingBoxElement(const int _idx) const
	{
		poca::core::BoundingBox bbox(FLT_MAX, FLT_MAX, m_zs.empty() ? 0.f : FLT_MAX, -FLT_MAX, -FLT_MAX, m_zs.empty() ? 0.f : -FLT_MAX);
		size_t ids[] = { m_triangles[_idx * 3], m_triangles[_idx * 3 + 1], m_triangles[_idx * 3 + 2] };
		for (unsigned int n = 0; n < 3; n++) {
			size_t index = ids[n];
			bbox[0] = m_xs[index] < bbox[0] ? m_xs[index] : bbox[0];
			bbox[1] = m_ys[index] < bbox[1] ? m_ys[index] : bbox[1];

			bbox[3] = m_xs[index] > bbox[3] ? m_xs[index] : bbox[3];
			bbox[4] = m_ys[index] > bbox[4] ? m_ys[index] : bbox[4];

			if (!m_zs.empty()) {
				bbox[2] = m_zs[index] < bbox[2] ? m_zs[index] : bbox[2];
				bbox[5] = m_zs[index] > bbox[5] ? m_zs[index] : bbox[5];
			}
		}
		return bbox;
	}

	poca::core::Vec3mf DelaunayTriangulation2D::computeBarycenterElement(const int _idx) const
	{
		poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
		size_t ids[] = { m_triangles[_idx * 3], m_triangles[_idx * 3 + 1], m_triangles[_idx * 3 + 2] };
		for (unsigned int n = 0; n < 3; n++) {
			size_t index = ids[n];
			centroid += poca::core::Vec3mf(m_xs[index], m_ys[index], m_zs.empty() ? 0.f : m_zs[index]);
		}
		centroid /= 3.f;
		return centroid;
	}

	void DelaunayTriangulation2D::getVerticesTriangle(const size_t _idx, std::vector <poca::core::Vec3mf>& _verts) const
	{
		size_t ids[] = { m_triangles[_idx * 3], m_triangles[_idx * 3 + 1], m_triangles[_idx * 3 + 2] };
		for (unsigned int n = 0; n < 3; n++) {
			size_t index = ids[n];
			_verts.push_back(poca::core::Vec3mf(m_xs[index], m_ys[index], m_zs.empty() ? 0.f : m_zs[index]));
		}
	}

	void DelaunayTriangulation2D::generateFaceSelectionFromLocSelection(const std::vector <bool>& _selecLocs, std::vector <bool>& _selecFaces)
	{
		_selecFaces.resize(nbFaces());
		for (size_t n = 0, cpt = 0; n < m_triangles.size(); n += 3, cpt++) {
			size_t i1 = m_triangles[n], i2 = m_triangles[n + 1], i3 = m_triangles[n + 2];
			_selecFaces[cpt] = _selecLocs[i1] && _selecLocs[i2] && _selecLocs[i3];
		}
	}

	void DelaunayTriangulation2D::generateFaceSelectionFromLocSelection(const std::vector <uint32_t>& _selecLocs, std::vector <uint32_t>& _selecFaces)
	{
		_selecFaces.resize(nbFaces());
		for (size_t n = 0, cpt = 0; n < m_triangles.size(); n += 3, cpt++) {
			size_t i1 = m_triangles[n], i2 = m_triangles[n + 1], i3 = m_triangles[n + 2];
			uint32_t o1 = _selecLocs[i1], o2 = _selecLocs[i2], o3 = _selecLocs[i3];
			_selecFaces[cpt] = (o1 == o2 && o2 == o3 && o1 != std::numeric_limits<uint32_t>::max()) ? o1 : std::numeric_limits<uint32_t>::max();
		}
	}

	void DelaunayTriangulation2D::generateFaceSelectionFromLocSelection(const std::vector <uint32_t>& _selecLocs, std::map <uint32_t, std::vector<uint32_t>>& _selecFaces)
	{
		uint32_t nbObjs = 0;
		for (uint32_t id : _selecLocs) {
			if (id == std::numeric_limits<std::uint32_t>::max()) continue;
			if (id > nbObjs) nbObjs = id;
		}
		for (auto n = 0; n < nbObjs; n++)
			_selecFaces[n] = std::vector <uint32_t>();
		const std::vector <uint32_t>& indices = m_neighbors.getFirstElements();
		for (size_t n = 0, cpt = 0; n < m_triangles.size(); n += 3, cpt++) {
			size_t i1 = m_triangles[n], i2 = m_triangles[n + 1], i3 = m_triangles[n + 2];
			uint32_t o1 = _selecLocs[i1], o2 = _selecLocs[i2], o3 = _selecLocs[i3];
			if (o1 == o2 && o2 == o3 && o1 != std::numeric_limits<uint32_t>::max())
				_selecFaces[o1].push_back(n / 3);
		}
	}

	bool DelaunayTriangulation2D::isPointInTriangle(const float _x, const float _y, const float _z, const uint32_t _idx) const
	{
		size_t ids[] = { m_triangles[_idx * 3], m_triangles[_idx * 3 + 1], m_triangles[_idx * 3 + 2] };
		poca::core::Vec3mf triVertices[] = { poca::core::Vec3mf(m_xs[ids[0]], m_ys[ids[0]], m_zs.empty() ? 0.f : m_zs[ids[0]]),
							poca::core::Vec3mf(m_xs[ids[1]], m_ys[ids[1]], m_zs.empty() ? 0.f : m_zs[ids[1]]),
							poca::core::Vec3mf(m_xs[ids[2]], m_ys[ids[2]], m_zs.empty() ? 0.f : m_zs[ids[2]]) };
		return poca::geometry::pointInTriangle(poca::core::Vec3mf(_x, _y, _z), triVertices[0], triVertices[1], triVertices[2]);
	}

	DelaunayTriangulation3D::DelaunayTriangulation3D(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector <float>& _zs, const std::vector <uint32_t>& _triangles, const std::vector <float>& _volumes, const poca::core::MyArrayUInt32& _neighbors, const uint32_t _nbFaces, Triangulation_3_inexact* _delau) : DelaunayTriangulationInterface(_xs, _ys, _zs, _triangles, _nbFaces)
	{
		m_internalDelaunay = _delau;
		m_neighbors = _neighbors;

		m_data["volume"] = poca::core::generateDataWithLog(_volumes);
		m_selection.resize(_volumes.size());
		setCurrentHistogramType("volume");
		forceRegenerateSelection();
	}

	DelaunayTriangulation3D::~DelaunayTriangulation3D()
	{
	}

	poca::core::BasicComponentInterface* DelaunayTriangulation3D::copy()
	{
		return new DelaunayTriangulation3D(*this);
	}

	void DelaunayTriangulation3D::generateTriangles(std::vector <poca::core::Vec3mf>& _triangles)
	{
		_triangles.resize(m_triangles.size());
		for (size_t n = 0; n < m_triangles.size(); n++) {
			size_t index = m_triangles[n];
			_triangles[n].set(m_xs[index], m_ys[index], m_zs[index]);
		}
	}

	void DelaunayTriangulation3D::getFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		_features.resize(m_triangles.size());

		size_t cpt = 0;
		for (size_t n = 0; n < m_nbCells; n++) {
			for (size_t i = 0; i < 12; i++)//4 triangles * 3 vertices per triangle
				_features[cpt++] = _selection[n] ? _values[n] : _notSelectedValue;
		}
	}

	void DelaunayTriangulation3D::generatePickingIndices(std::vector <float>& _ids) const
	{
		_ids.resize(m_triangles.size());
		int curId = 0;
		std::generate(_ids.begin(), _ids.end(), [&, cpt = 0]() mutable { if (cpt % 12 == 0) curId++; cpt++; return curId; });
	}

	poca::core::BoundingBox DelaunayTriangulation3D::computeBoundingBoxElement(const int _idx) const
	{
		poca::core::BoundingBox bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
		size_t first = _idx * 12; // 1 tetrahedron = 4 triangles * 3 vertices
		for (unsigned int n = first; n < first + 12; n++) {
			size_t index = m_triangles[n];
			bbox[0] = m_xs[index] < bbox[0] ? m_xs[index] : bbox[0];
			bbox[1] = m_ys[index] < bbox[1] ? m_ys[index] : bbox[1];
			bbox[2] = m_zs[index] < bbox[2] ? m_zs[index] : bbox[2];

			bbox[3] = m_xs[index] > bbox[3] ? m_xs[index] : bbox[3];
			bbox[4] = m_ys[index] > bbox[4] ? m_ys[index] : bbox[4];
			bbox[5] = m_zs[index] > bbox[5] ? m_zs[index] : bbox[5];
		}
		return bbox;
	}

	poca::core::Vec3mf DelaunayTriangulation3D::computeBarycenterElement(const int _idx) const
	{
		poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
		size_t first = _idx * 12; // 1 tetrahedron = 4 triangles * 3 vertices
		for (unsigned int n = first; n < first + 12; n++) {
			size_t index = m_triangles[n];
			centroid += poca::core::Vec3mf(m_xs[index], m_ys[index], m_zs[index]);
		}
		centroid /= 12.f;
		return centroid;
	}

	void DelaunayTriangulation3D::getVerticesTriangle(const size_t _idx, std::vector <poca::core::Vec3mf>& _verts) const
	{
		size_t first = _idx * 12; // 1 tetrahedron = 4 triangles * 3 vertices
		for (unsigned int n = first; n < first + 12; n++){//; n += 4) {
			size_t index = m_triangles[n];
			_verts.push_back(poca::core::Vec3mf(m_xs[index], m_ys[index], m_zs[index]));
		}
		
	}

	void DelaunayTriangulation3D::generateFaceSelectionFromLocSelection(const std::vector <bool>& _selecLocs, std::vector <bool>& _selecFaces)
	{
		_selecFaces.resize(nbFaces());
		const std::vector <uint32_t>& indices = m_neighbors.getFirstElements();
		for (uint32_t n = 0; n < nbFaces(); n++) {
			uint32_t index = indices[n];
			uint32_t i1 = m_triangles[3 * index],
				i2 = m_triangles[3 * index + 3 * 1],
				i3 = m_triangles[3 * index + 3 * 2],
				i4 = m_triangles[3 * index + 3 * 3];
			_selecFaces[n] = _selecLocs[i1] && _selecLocs[i2] && _selecLocs[i3] && _selecLocs[i4];
		}
	}

	void DelaunayTriangulation3D::generateFaceSelectionFromLocSelection(const std::vector <uint32_t>& _selecLocs, std::vector <uint32_t>& _selecFaces)
	{
		_selecFaces.resize(nbFaces());
		const std::vector <uint32_t>& indices = m_neighbors.getFirstElements();
		for (uint32_t n = 0; n < nbFaces(); n++) {
			uint32_t index = indices[n];
			uint32_t i1 = m_triangles[3 * index],
				i2 = m_triangles[3 * index + 3 * 1],
				i3 = m_triangles[3 * index + 3 * 2],
				i4 = m_triangles[3 * index + 3 * 3];
			uint32_t o1 = _selecLocs[i1], 
				o2 = _selecLocs[i2], 
				o3 = _selecLocs[i3],
				o4 = _selecLocs[i4];
			_selecFaces[n] = (o1 == o2 && o2 == o3 && o3 == o4 && o1 != std::numeric_limits<uint32_t>::max()) ? o1 : std::numeric_limits<uint32_t>::max();
		}
	}

	void DelaunayTriangulation3D::generateFaceSelectionFromLocSelection(const std::vector <uint32_t>& _selecLocs, std::map <uint32_t, std::vector<uint32_t>>& _selecFaces)
	{
		uint32_t nbObjs = 0;
		for (uint32_t id : _selecLocs) {
			if (id == std::numeric_limits<std::uint32_t>::max()) continue;
			if (id > nbObjs) nbObjs = id;
		}
		for (auto n = 0; n < nbObjs; n++)
			_selecFaces[n] = std::vector <uint32_t>();
		const std::vector <uint32_t>& indices = m_neighbors.getFirstElements();
		for (uint32_t n = 0; n < nbFaces(); n++) {
			uint32_t index = n * 12;
			uint32_t i1 = m_triangles[index],
				i2 = m_triangles[index + 3 * 1],
				i3 = m_triangles[index + 3 * 2],
				i4 = m_triangles[index + 3 * 3];
			uint32_t o1 = _selecLocs[i1],
				o2 = _selecLocs[i2],
				o3 = _selecLocs[i3],
				o4 = _selecLocs[i4];
			if (o1 == o2 && o2 == o3 && o3 == o4 && o1 != std::numeric_limits<uint32_t>::max())
				_selecFaces[o1].push_back(n);
		}
	}

	uint32_t DelaunayTriangulation3D::indexTriangleOfPoint(const float _x, const float _y, const float _z, const uint32_t _idx)
	{
		Point_3_inexact p(_x, _y, _z);
		Triangulation_3_inexact::Cell_handle cell = m_internalDelaunay->locate(p);
		if (cell == NULL || cell->info() == -1) return std::numeric_limits<std::uint32_t>::max();
		return cell->info();
	}
}

