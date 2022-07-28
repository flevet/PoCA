/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagram.cpp
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
#include <qmath.h>

#include <General/MyData.hpp>

#include "VoronoiDiagram.hpp"
#include "BasicComputation.hpp"

namespace poca::geometry {
	VoronoiDiagram::VoronoiDiagram(const float* _xs, const float* _ys, const float* _zs, 
		const uint32_t _nbCells, const std::vector <uint32_t>& _neighbors,
		const std::vector <uint32_t>& _firstsNeighbors, KdTree_DetectionPoint* _kdtree,
		DelaunayTriangulationInterface* _delau) :BasicComponent("VoronoiDiagram"), m_xs(_xs), m_ys(_ys), m_zs(_zs), m_nbCells(_nbCells), m_neighbors(_neighbors, _firstsNeighbors), m_deleteKdTree(false), m_delaunay(_delau)
	{
		if (_kdtree)
			m_kdTree = _kdtree;
		else {
			m_deleteKdTree = true;
			poca::core::DetectionPointCloud pointCloud;
			pointCloud.resize(m_nbCells);
			if (m_zs != NULL)
				for (size_t n = 0; n < m_nbCells; n++)
					pointCloud.m_pts[n].set(_xs[n], _ys[n], _zs[n]);
			else
				for (size_t n = 0; n < m_nbCells; n++)
					pointCloud.m_pts[n].set(_xs[n], _ys[n], 0.);
			m_kdTree = new KdTree_DetectionPoint(3, pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
			m_kdTree->buildIndex();
		}
	}

	VoronoiDiagram::~VoronoiDiagram()
	{
		if (m_deleteKdTree)
			delete m_kdTree;
	}

	const float VoronoiDiagram::averageMeanNbLocs() const
	{
		const poca::core::BoundingBox& bbox = boundingBox();
		float w = bbox[3] - bbox[0], h = bbox[4] - bbox[1], t = bbox[5] - bbox[2];
		float val = dimension() == 3 ? w * h * t : w * h;
		return val / (float)m_nbCells;
		
	}

	const float VoronoiDiagram::averageDensity() const
	{
		const poca::core::BoundingBox& bbox = boundingBox();
		float w = bbox[3] - bbox[0], h = bbox[4] - bbox[1], t = bbox[5] - bbox[2];
		float val = dimension() == 3 ? w * h * t : w * h;
		return (float)m_nbCells / val;
	}

	VoronoiDiagram2D::VoronoiDiagram2D(const poca::core::Vec3md* _cells, const uint32_t _nbEdges, const std::vector<uint32_t>& _firsts, const std::vector<uint32_t>& _neighs, const float *_xs, const float * _ys, const float* _zs, KdTree_DetectionPoint* _kdtree, DelaunayTriangulationInterface * _delau) : VoronoiDiagram(_xs, _ys, _zs, _firsts.size() - 1, _neighs, _firsts, _kdtree, _delau)
	{
		std::copy(&_cells[0], &_cells[_nbEdges], std::back_inserter(m_cells));
		//Create area feature
		const std::vector <uint32_t>& firsts = m_neighbors.getFirstElements();
		const std::vector <uint32_t>& neighs = m_neighbors.getData();
		std::vector <float> areas(nbFaces(), 0.f), densities(nbFaces(), 0.f);
		float maxArea = -FLT_MAX;
		for (size_t n = 0; n < areas.size(); n++) {
			size_t first = firsts[n], nbs = firsts[n + 1] - firsts[n];
			if (nbs > 0) {
				areas[n] = computePolygonArea(m_cells.data() + first, nbs);
			}
			if (areas[n] > maxArea) maxArea = areas[n];
		}
		for (size_t n = 0; n < areas.size(); n++)
			if (areas[n] == 0.f) {
				areas[n] = maxArea;
			}
		for (size_t n = 0, cpt = 0; n < nbFaces(); n++) {
			float sumArea = areas[n], nbsTot = 1.f;
			uint32_t first = firsts[n], nbs = firsts[n + 1] - firsts[n];
			for (uint32_t i = 0; i < nbs; i++) {
				uint32_t index = first + i, neigh = neighs[index];
				if (neigh != std::numeric_limits<std::uint32_t>::max()) {
					sumArea += areas[neigh];
					nbsTot += 1.f;
				}
			}
			densities[n] = nbsTot / sumArea;
		}
		m_data["area"] = new poca::core::MyData(areas);
		m_data["density"] = new poca::core::MyData(densities);

		std::vector <float> ids(areas.size());
		std::iota(std::begin(ids), std::end(ids), 1);
		m_data["id"] = new poca::core::MyData(ids);

		m_selection.resize(areas.size());
		setCurrentHistogramType("density");
		forceRegenerateSelection();
	}

	VoronoiDiagram2D::~VoronoiDiagram2D()
	{
	}

	poca::core::BasicComponent* VoronoiDiagram2D::copy()
	{
		return new VoronoiDiagram2D(*this);
	}

	void VoronoiDiagram2D::generateTriangles(std::vector <poca::core::Vec3mf>& _triangles)
	{
		const std::vector <uint32_t>& firsts = m_neighbors.getFirstElements();
		_triangles.resize(firsts.back() * 3);
		for (uint32_t n = 0, cpt = 0; n < nbFaces(); n++) {
			uint32_t first = firsts[n], nbs = firsts[n + 1] - firsts[n];
			poca::core::Vec3md centroid;
			double divisor = nbs;
			for (uint32_t i = 0; i < nbs; i++) {
				uint32_t index = first + i;
				centroid += (m_cells[index] / divisor);
			}
			for (uint32_t i = 0; i < nbs; i++) {
				uint32_t cur = first + i, next = first + ((i + 1) % nbs);
				_triangles[cpt++].set((float)centroid[0], (float)centroid[1], (float)centroid[2]);
				_triangles[cpt++].set((float)m_cells[cur][0], (float)m_cells[cur][1], (float)m_cells[cur][2]);
				_triangles[cpt++].set((float)m_cells[next][0], (float)m_cells[next][1], (float)m_cells[next][2]);
			}
		}
	}

	void VoronoiDiagram2D::generateLines(std::vector <poca::core::Vec3mf>& _lines)
	{
		const std::vector <uint32_t>& firsts = m_neighbors.getFirstElements();
		_lines.resize(firsts.back() * 2);
		for (uint32_t n = 0, cpt = 0; n < nbFaces(); n++) {
			uint32_t first = firsts[n], nbs = firsts[n + 1] - firsts[n];
			for (uint32_t i = 0; i < nbs; i++) {
				uint32_t cur = first + i, next = first + ((i + 1) % nbs);
				_lines[cpt++].set((float)m_cells[cur][0], (float)m_cells[cur][1], (float)m_cells[cur][2]);
				_lines[cpt++].set((float)m_cells[next][0], (float)m_cells[next][1], (float)m_cells[next][2]);
			}
		}
	}


	void VoronoiDiagram2D::getFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue, const bool _forLine) const
	{
		const std::vector <uint32_t>& firsts = m_neighbors.getFirstElements();
		uint32_t multiplierPrimitive = _forLine ? 2 : 3;//2 for lines, 3 for triangles
		_features.resize(firsts.back() * multiplierPrimitive);
		unsigned int cpt = 0;
		for (unsigned int n = 0; n < nbFaces(); n++) {
			uint32_t first = firsts[n], nbs = firsts[n + 1] - firsts[n];
			float value = _selection[n] ? _values[n] : _notSelectedValue;
			for (uint32_t i = 0; i < nbs; i++)
				for(uint32_t j = 0; j < multiplierPrimitive; j++)
					_features[cpt++] = value;
		}
	}

	void VoronoiDiagram2D::generatePickingIndices(std::vector <float>& _ids) const
	{
		const std::vector <uint32_t>& firsts = m_neighbors.getFirstElements();
		_ids.resize(firsts.back() * 3);
		for (uint32_t n = 0, cpt = 0; n < nbFaces(); n++) {
			uint32_t first = firsts[n], nbs = firsts[n + 1] - firsts[n];
			for (uint32_t i = 0; i < nbs; i++) {
				_ids[cpt++] = n + 1;
				_ids[cpt++] = n + 1;
				_ids[cpt++] = n + 1;
			}
		}
	}

	poca::core::BoundingBox VoronoiDiagram2D::computeBoundingBoxElement(const int _idx) const
	{
		const std::vector <uint32_t>& firsts = m_neighbors.getFirstElements();
		poca::core::BoundingBox bbox(FLT_MAX, FLT_MAX, m_zs == NULL ? 0.f : FLT_MAX, -FLT_MAX, -FLT_MAX, m_zs == NULL ? 0.f : -FLT_MAX);
		uint32_t first = firsts[_idx], nbs = firsts[_idx + 1] - firsts[_idx];
		for (uint32_t n = firsts[_idx]; n < firsts[_idx + 1]; n++) {
			bbox[0] = m_cells[n][0] < bbox[0] ? m_cells[n][0] : bbox[0];
			bbox[1] = m_cells[n][1] < bbox[1] ? m_cells[n][1] : bbox[1];

			bbox[3] = m_cells[n][0] > bbox[3] ? m_cells[n][0] : bbox[3];
			bbox[4] = m_cells[n][1] > bbox[4] ? m_cells[n][1] : bbox[4];

			if (m_zs != NULL) {
				bbox[2] = m_cells[n][2] < bbox[2] ? m_cells[n][2] : bbox[2];
				bbox[5] = m_cells[n][2] > bbox[5] ? m_cells[n][2] : bbox[5];
			}
		}
		return bbox;
	}

	poca::core::Vec3mf VoronoiDiagram2D::computeBarycenterElement(const int _idx) const
	{
		poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
		const std::vector <uint32_t>& firsts = m_neighbors.getFirstElements();
		float nbs = firsts[_idx + 1] - firsts[_idx];
		for (uint32_t n = firsts[_idx]; n < firsts[_idx + 1]; n++)
			centroid += poca::core::Vec3mf(m_cells[n][0], m_cells[n][1], 0.f) / nbs;
		return centroid;
	}

	//Bad implmentation: use proper walking through triangulation
	uint32_t VoronoiDiagram2D::indexTriangleOfPoint(const float _x, const float _y, const float _z) const
	{
		const std::size_t num_results = 1;
		uint32_t triangleIndex = std::numeric_limits<std::uint32_t>::max();
		std::vector<size_t> ret_index(num_results);
		std::vector<double> out_dist_sqr(num_results);
		const double queryPt[3] = { _x, _y, _z };
		m_kdTree->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
		triangleIndex = m_delaunay->indexTriangleOfPoint(_x, _y, _z, ret_index[0]);
		return triangleIndex;
	}

	VoronoiDiagram2DOnSphere::VoronoiDiagram2DOnSphere(const std::vector<poca::core::Vec3md>& _cells, const uint32_t _nbCells, const std::vector<uint32_t>& _firsts, const std::vector<uint32_t>& _neighs, const std::vector <poca::core::Vec3mf>& _normals, const float* _xs, const float* _ys, const float* _zs, const poca::core::Vec3mf& _centroid, const float _radius, KdTree_DetectionPoint* _kdtree, DelaunayTriangulationInterface* _delau) : VoronoiDiagram2D(_cells.data(), _nbCells, _firsts, _neighs, _xs, _ys, _zs, _kdtree, _delau), m_normals(_normals), m_centroid(_centroid), m_radius(_radius)
	{
	}

	VoronoiDiagram2DOnSphere::~VoronoiDiagram2DOnSphere()
	{
	}

	void VoronoiDiagram2DOnSphere::generateLinesNormals(std::vector <poca::core::Vec3mf>& _normals)
	{
		const std::vector <uint32_t>& firsts = m_neighbors.getFirstElements();
		_normals.resize(firsts.back() * 2);
		for (uint32_t n = 0, cpt = 0; n < nbFaces(); n++) {
			uint32_t first = firsts[n], nbs = firsts[n + 1] - firsts[n];
			for (uint32_t i = 0; i < nbs; i++) {
				uint32_t cur = first + i, next = first + ((i + 1) % nbs);
				_normals[cpt++] = m_normals[n];
				_normals[cpt++] = m_normals[n];
			}
		}
	}

	const float VoronoiDiagram2DOnSphere::averageMeanNbLocs() const
	{
		float surface = 4 * M_PI * pow(m_radius, 2.f);
		return surface / (float)m_nbCells;

	}

	const float VoronoiDiagram2DOnSphere::averageDensity() const
	{
		float surface = 4 * M_PI * pow(m_radius, 2.f);
		return (float)m_nbCells / surface;
	}

	VoronoiDiagram3D::VoronoiDiagram3D(const uint32_t _nbCells,
		const std::vector <uint32_t>& _neighbors,
		const std::vector <uint32_t>& _firstsNeighbors,
		const std::vector <float>& _volumes, 
		const float* _xs, const float* _ys, const float* _zs, 
		KdTree_DetectionPoint* _kdtree, DelaunayTriangulationInterface* _delau) : VoronoiDiagram(_xs, _ys, _zs, _nbCells, _neighbors, _firstsNeighbors, _kdtree, _delau)
	{
		double nbPs = nbFaces();
		unsigned int nbForUpdate = nbPs / 100., cptTimer = 0;
		if (nbForUpdate == 0) nbForUpdate = 1;
		std::printf("Computing Voronoi density: %.2f %%", (0. / nbPs * 100.));
		std::vector <float> densities(nbFaces(), 0.f);
		for (size_t n = 0, cpt = 0; n < nbFaces(); n++) {
			float sumVol = _volumes[n], nbsTot = 1.f;
			for (size_t index = _firstsNeighbors[n]; index < _firstsNeighbors[n + 1]; index++) {
				size_t neigh = _neighbors[index];
				if (neigh != std::numeric_limits<std::size_t>::max()) {
					sumVol += _volumes[neigh];
					nbsTot += 1.f;
				}
			}
			densities[n] = nbsTot / sumVol;
			if (cptTimer++ % nbForUpdate == 0) 	std::printf("\rComputing Voronoi density: %.2f %%", ((double)cptTimer / nbPs * 100.));
		}
		std::cout << std::endl;
		m_data["volume"] = new poca::core::MyData(_volumes);
		m_data["density"] = new poca::core::MyData(densities);
		std::vector <float> ids(_volumes.size());
		std::iota(std::begin(ids), std::end(ids), 1);
		m_data["id"] = new poca::core::MyData(ids);

		m_selection.resize(_volumes.size());
		setCurrentHistogramType("density");
		forceRegenerateSelection();
	}

	VoronoiDiagram3D::VoronoiDiagram3D(const uint32_t _nbCells,
		const std::vector <uint32_t>& _neighbors,
		const std::vector <uint32_t>& _firstsNeighbors,
		const std::vector <poca::core::Vec3mf>& _triangles,
		const std::vector <uint32_t>& _firstTriangleCell,
		const std::vector <float>& _volumes,
		const float* _xs, const float* _ys, const float* _zs, 
		KdTree_DetectionPoint* _kdtree, DelaunayTriangulationInterface* _delau) : VoronoiDiagram(_xs, _ys, _zs, _nbCells, _neighbors, _firstsNeighbors, _kdtree, _delau),
																	m_cells(_triangles), m_firstCells(_firstTriangleCell)
	{
		std::vector <float> densities(nbFaces(), 0.f);
		for (size_t n = 0, cpt = 0; n < nbFaces(); n++) {
			float sumVol = _volumes[n], nbsTot = 1.f;
			for (size_t index = _firstsNeighbors[n]; index < _firstsNeighbors[n + 1]; index++) {
				size_t neigh = _neighbors[index];
				if (neigh != std::numeric_limits<std::size_t>::max()) {
					sumVol += _volumes[neigh];
					nbsTot += 1.f;
				}
			}
			densities[n] = nbsTot / sumVol;
		}

		m_data["volume"] = new poca::core::MyData(_volumes);
		m_data["density"] = new poca::core::MyData(densities);
		std::vector <float> ids(_volumes.size());
		std::iota(std::begin(ids), std::end(ids), 1);
		m_data["id"] = new poca::core::MyData(ids);
		m_selection.resize(_volumes.size());
		setCurrentHistogramType("density");
		forceRegenerateSelection();
	}

	VoronoiDiagram3D::~VoronoiDiagram3D()
	{
	}

	poca::core::BasicComponent* VoronoiDiagram3D::copy()
	{
		return new VoronoiDiagram3D(*this);
	}

	void VoronoiDiagram3D::generateTriangles(std::vector <poca::core::Vec3mf>& _triangles)
	{
		if (m_cells.empty()) return;
		_triangles.resize(m_cells.size());
		std::copy(m_cells.begin(), m_cells.end(), _triangles.begin());
	}

	void VoronoiDiagram3D::generateLines(std::vector <poca::core::Vec3mf>& _lines)
	{
	}

	void VoronoiDiagram3D::getFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue, const bool _forLine) const
	{
		if (m_cells.empty()) return;
		_features.resize(m_cells.size());
		size_t cpt = 0;
		for (size_t n = 0; n < m_nbCells; n++) {
			unsigned int size = m_firstCells[n + 1] - m_firstCells[n];
			for (size_t i = 0; i < size; i++)
				_features[cpt++] = _selection[n] ? _values[n] : _notSelectedValue;
		}
	}

	void VoronoiDiagram3D::generatePickingIndices(std::vector <float>& _ids) const
	{
		if (m_cells.empty()) return;
		_ids.resize(m_cells.size());

		size_t cpt = 0;
		for (size_t n = 0; n < m_nbCells; n++) {
			uint32_t size = m_firstCells[n + 1] - m_firstCells[n];
			for (size_t i = 0; i < size; i++)//4 triangles * 3 vertices per triangle
				_ids[cpt++] = n + 1;
		}
	}

	poca::core::BoundingBox VoronoiDiagram3D::computeBoundingBoxElement(const int _idx) const
	{
		poca::core::BoundingBox bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
		if (m_cells.empty()) return bbox;
		for (unsigned int n = m_firstCells[_idx]; n < m_firstCells[_idx + 1]; n++) {
			bbox[0] = m_cells[n].x() < bbox[0] ? m_cells[n].x() : bbox[0];
			bbox[1] = m_cells[n].y() < bbox[1] ? m_cells[n].y() : bbox[1];
			bbox[2] = m_cells[n].z() < bbox[2] ? m_cells[n].z() : bbox[2];

			bbox[3] = m_cells[n].x() > bbox[3] ? m_cells[n].x() : bbox[3];
			bbox[4] = m_cells[n].y() > bbox[4] ? m_cells[n].y() : bbox[4];
			bbox[5] = m_cells[n].z() > bbox[5] ? m_cells[n].z() : bbox[5];
		}
		return bbox;
	}

	poca::core::Vec3mf VoronoiDiagram3D::computeBarycenterElement(const int _idx) const
	{
		poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
		const std::vector <uint32_t>& firsts = m_neighbors.getFirstElements();
		float nbs = firsts[_idx + 1] - firsts[_idx];
		for (uint32_t n = firsts[_idx]; n < firsts[_idx + 1]; n++)
			centroid += m_cells[n] / nbs;
		return centroid;
	}

	uint32_t VoronoiDiagram3D::indexTriangleOfPoint(const float _x, const float _y, const float _z) const
	{
		return m_delaunay->indexTriangleOfPoint(_x, _y, _z);
	}
}

