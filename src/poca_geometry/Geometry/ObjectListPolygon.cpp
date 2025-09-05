/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListPolygon.cpp
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

#include <algorithm>
#include <execution>
#include <CGAL/centroid.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Triangulation_conformer_2.h>

#include <General/MyData.hpp>
#include <General/BasicComponent.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/ROIInterface.hpp>
#include <General/Misc.h>

#include "ObjectListPolygon.hpp"
#include "BasicComputation.hpp"
#include "DelaunayTriangulation.hpp"
#include "../Interfaces/ObjectFeaturesFactoryInterface.hpp"

namespace poca::geometry {
	ObjectListPolygon::ObjectListPolygon(const float* _xs, const float* _ys, const float* _zs, 
		const std::vector <std::vector<Polygon_2>>& _polygons, const std::vector <uint32_t>& _locsAllObjects, 
		const std::vector <uint32_t>& _firstsLocs, const std::vector <uint32_t>& _linkTriangulationFacesToObjects) : ObjectListInterface("ObjectListPolygon", _locsAllObjects, _firstsLocs), m_polygons(_polygons), m_xs(_xs), m_ys(_ys), m_zs(_zs), m_linkTriangulationFacesToObjects(_linkTriangulationFacesToObjects)
	{
		generateFromPolygons();
	}

	ObjectListPolygon::ObjectListPolygon(const std::vector <std::vector<Polygon_2>>& _polygons) : ObjectListInterface("ObjectListPolygon"), m_polygons(_polygons)
	{
		generateFromPolygons();
	}

	ObjectListPolygon::ObjectListPolygon(std::vector <std::vector<Polygon_2>>::const_iterator _begin, std::vector <std::vector<Polygon_2>>::const_iterator _end) : ObjectListInterface("ObjectListPolygon"), m_polygons(_begin, _end)
	{
		generateFromPolygons();
	}

	ObjectListPolygon::ObjectListPolygon(const std::vector <std::vector <std::vector <poca::core::Vec3mf>>>& _allSegments) : ObjectListInterface("ObjectListPolygon")
	{
		m_polygons.resize(_allSegments.size());

		uint32_t curObj = 0, curOutline = 0;
		for (const auto& segments : _allSegments) {
			for (const auto& segment : segments) {
				std::vector < Point_2 > ptsTmp;
				for (const auto& pt : segment)
					ptsTmp.push_back(Point_2(pt.x(), pt.y()));
				m_polygons[curObj].emplace_back(ptsTmp.begin(), ptsTmp.begin() + ptsTmp.size());
				/*for (const auto& point : m_polygons[curObj].back().container()) {
					std::cout << point << std::endl;
				}*/
				if (m_polygons[curObj].back().is_clockwise_oriented()) {
					std::cout << "Object " << curObj << ", cur outline " << m_polygons[curObj].size() << std::endl;
					m_polygons[curObj].back() = Polygon_2(ptsTmp.rbegin(), ptsTmp.rbegin() + ptsTmp.size());
				}
			}
			curObj++;
		}

		generateFromPolygons();
	}

	void ObjectListPolygon::generateFromPolygons()
	{
		std::vector <float> area;

		m_centroids.resize(m_polygons.size());
		m_bboxMeshes.resize(m_polygons.size());

		m_bbox.set(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0.f, std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), 0.f);
		uint32_t curObj = 0;
		for (const auto& polygons : m_polygons) {
			const auto& polygon = polygons.front();
			auto centroid = CGAL::centroid(polygon.vertices_begin(), polygon.vertices_end());
			auto bbox = CGAL::bounding_box(polygon.vertices_begin(), polygon.vertices_end());
			m_centroids[curObj].set(centroid.x(), centroid.y(), 0.f);
			m_bboxMeshes[curObj++].set(bbox.xmin(), bbox.ymin(), 0.f, bbox.xmax(), bbox.ymax(), 0.f);
			m_bbox.addPointBBox(bbox.xmin(), bbox.ymin(), 0.f);
			m_bbox.addPointBBox(bbox.xmax(), bbox.ymax(), 0.f);
		}

		if (m_locs.empty()) {
			std::vector <uint32_t> points, nbPts{ 0 }; //_mesh.number_of_vertices()
			uint32_t curObj = 0;
			for (const auto& polygons : m_polygons) {
				const auto& polygon = polygons.front();
				for (const auto& p : polygon.container()) {
					auto x = p.x(), y = p.y();
					m_xsDuplicate.push_back(x);
					m_ysDuplicate.push_back(y);
				}
				nbPts.push_back(m_xsDuplicate.size());
			}
			m_xs = m_xsDuplicate.data();
			m_ys = m_ysDuplicate.data();
			m_zs = NULL;

			points.resize(nbPts.back());
			std::iota(std::begin(points), std::end(points), 0);
			m_locs.initialize(points, nbPts);
			m_outlineLocs = m_locs;
		}

		std::vector <poca::core::Vec3mf> outlines;
		std::vector <uint32_t> nbSegments{ 0 }; //_mesh.number_of_vertices()
		for (const auto& polygons : m_polygons) {
			//std::cout << "****************************************" << std::endl;
			for (const auto& polygon : polygons) {
				//std::cout << "Aera = " << fabs(polygon.area()) << ", # verts = " << polygon.size() << std::endl;
				const auto& points = polygon.container();
				std::size_t n = points.size();

				for (std::size_t i = 0; i < n; ++i) {
					const auto& curr = points[i];
					const auto& next = points[(i + 1) % n];  // wrap around

					outlines.emplace_back(curr.x(), curr.y(), 0.f);
					outlines.emplace_back(next.x(), next.y(), 0.f);
				}
			}
			nbSegments.push_back(outlines.size());
		}
		m_outlines.initialize(outlines, nbSegments);
		//std::cout << "****************************************" << std::endl;

		std::vector <poca::core::Vec3mf> triangles;
		std::vector <uint32_t> nbTriangles{ 0 }; //_mesh.number_of_vertices()
		if (false) {
			for (const auto& polygons : m_polygons) {
				const auto& polygon = polygons.size() >= 2 ? polygons[1] : polygons[0];
					Constrained_Delaunay_triangulation_2_tag cdt;
					auto insert_polygon = [&cdt](const Polygon_2& poly) {
						for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it) {
							auto next = std::next(it);
							if (next == poly.vertices_end()) next = poly.vertices_begin();
							cdt.insert_constraint(*it, *next);
						}
						};

					insert_polygon(polygon);

					size_t currentIndex = 0;
					for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
						Point_2 p0 = fit->vertex(0)->point();
						Point_2 p1 = fit->vertex(1)->point();
						Point_2 p2 = fit->vertex(2)->point();

						// Use centroid to check whether triangle is inside area of interest
						Point_2 centroid((p0.x() + p1.x() + p2.x()) / 3,
							(p0.y() + p1.y() + p2.y()) / 3);

						bool inside = polygon.bounded_side(centroid) == CGAL::ON_BOUNDED_SIDE;

						if (inside) {
							triangles.emplace_back(p0.x(), p0.y(), 0.f);
							triangles.emplace_back(p1.x(), p1.y(), 0.f);
							triangles.emplace_back(p2.x(), p2.y(), 0.f);
							fit->info().m_index = currentIndex++;
						}
						fit->info().m_tag = inside ? poca::geometry::INSIDE : poca::geometry::OUTSIDE;
					}
				nbTriangles.push_back(triangles.size());
				//std::cout << "Polygon, nb triangles " << triangles.size() << std::endl;
			}
		}
		else {
			for (const auto& polygons : m_polygons) {
				//std::cout << "****************************************\n# polygons " << polygons.size() << std::endl;

				for(const auto& pol : polygons)
					std::cout << "Polygon, # locs " << pol.size() << ", area " << pol.area() << std::endl;

				Polygon_with_holes_2_inexact pwh(polygons.front());
				for (auto n = 1; n < polygons.size(); n++)
					pwh.add_hole(polygons[n]);

				//std::cout << "-";
				m_cdts.push_back(Constrained_Delaunay_triangulation_2_tag());
				auto& cdt = m_cdts.back();
				//Constrained_Delaunay_triangulation_2 cdt;

				//std::cout << "-";
				auto insert_polygon = [&cdt](const Polygon_2& poly) {
					for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it) {
						auto next = std::next(it);
						if (next == poly.vertices_end()) next = poly.vertices_begin();
						cdt.insert_constraint(*it, *next);
					}
					};
				//std::cout << "-";

				insert_polygon(pwh.outer_boundary());
				for (auto hit = pwh.holes_begin(); hit != pwh.holes_end(); ++hit) {
					insert_polygon(*hit);
				}
				//std::cout << "-";

				// Optional: refine triangulation
				//CGAL::make_conforming_Delaunay_2(cdt);
				//CGAL::make_conforming_Gabriel_2(cdt);

				// 5. Collect valid triangles (inside outer, outside holes)
				//std::cout << "\nnumber of faces " << cdt.number_of_faces() << ", # vertices = " << cdt.number_of_vertices() << std::endl;
				size_t currentIndex = 0;
				for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
					Point_2 p0 = fit->vertex(0)->point();
					Point_2 p1 = fit->vertex(1)->point();
					Point_2 p2 = fit->vertex(2)->point();

					// Use centroid to check whether triangle is inside area of interest
					Point_2 centroid((p0.x() + p1.x() + p2.x()) / 3,
						(p0.y() + p1.y() + p2.y()) / 3);

					/*bool inside = pwh.outer_boundary().bounded_side(centroid) == CGAL::ON_BOUNDED_SIDE;

					if (inside) {
						triangles.emplace_back(p0.x(), p0.y(), 0.f);
						triangles.emplace_back(p1.x(), p1.y(), 0.f);
						triangles.emplace_back(p2.x(), p2.y(), 0.f);
						fit->info().m_index = currentIndex++;
					}
					fit->info().m_tag = inside ? poca::geometry::INSIDE : poca::geometry::OUTSIDE;*/



					// Check that centroid is INSIDE outer boundary and OUTSIDE all holes
					if (pwh.outer_boundary().bounded_side(centroid) == CGAL::ON_BOUNDED_SIDE) {
						bool inside_hole = false;
						for (auto hit = pwh.holes_begin(); hit != pwh.holes_end(); ++hit) {
							if (hit->bounded_side(centroid) == CGAL::ON_BOUNDED_SIDE) {
								inside_hole = true;
								break;
							}
						}
						if (!inside_hole) {
							triangles.emplace_back(p0.x(), p0.y(), 0.f);
							triangles.emplace_back(p1.x(), p1.y(), 0.f);
							triangles.emplace_back(p2.x(), p2.y(), 0.f);
							fit->info().m_index = currentIndex++;
						}
						fit->info().m_tag = inside_hole ? poca::geometry::OUTSIDE : poca::geometry::INSIDE;
					}
					else
						fit->info().m_tag = poca::geometry::OUTSIDE;
				}
				nbTriangles.push_back(triangles.size());
				//std::cout << "Polygon, nb triangles " << triangles.size() << std::endl;
				//for (const auto& pol : polygons) {
				//	std::cout << "\t area -> " << fabs(pol.area()) << std::endl;
				//}
			}
		}
		m_triangles.initialize(triangles, nbTriangles);

		//Create area feature
		std::vector <float> nbLocs(m_locs.nbElements(), 0.f);
		for (size_t i = 0; i < m_triangles.nbElements(); i++)
			nbLocs[i] = m_locs.nbElementsObject(i);
		const poca::core::MyArrayUInt32& localizations = m_locs;// m_outlineLocs;
		ObjectFeaturesFactoryInterface* factory = createObjectFeaturesFactory();
		std::vector <float> sizes(m_locs.nbElements()), resPCA(factory->nbFeaturesPCA(false));
		std::vector <float> circ(m_locs.nbElements()), areas(m_locs.nbElements());
		m_axis.resize(m_locs.nbElements());
		for (size_t n = 0; n < m_locs.nbElements(); n++) {
			float* ptr = &resPCA[0];
			factory->computePCA(m_locs, n, m_xs, m_ys, m_zs, ptr);
			sizes[n] = (resPCA[8] + resPCA[9]) / 2.f;
			circ[n] = resPCA[7];
			areas[n] = fabs(m_polygons[n].front().area());
			for(auto i = 1; i < m_polygons[n].size(); i++)
				areas[n] -= fabs(m_polygons[n][i].area());;
		}
		delete factory;

		std::vector <float> ids(m_locs.nbElements());
		std::iota(std::begin(ids), std::end(ids), 1);

		m_data["area"] = poca::core::generateDataWithLog(areas);
		m_data["nbLocs"] = poca::core::generateDataWithLog(nbLocs);
		m_data["size"] = poca::core::generateDataWithLog(sizes);
		m_data["circ"] = poca::core::generateDataWithLog(circ);
		m_data["id"] = poca::core::generateDataWithLog(ids);

		m_selection.resize(areas.size());
		setCurrentHistogramType("area");
		forceRegenerateSelection();
	}

	ObjectListPolygon::~ObjectListPolygon()
	{
	}

	poca::core::BasicComponentInterface* ObjectListPolygon::copy()
	{
		return new ObjectListPolygon(*this);
	}

	poca::core::BasicComponentInterface* ObjectListPolygon::copy(const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		if (_ROIs.empty())
			return copy();

		std::vector <std::vector <std::vector <poca::core::Vec3mf>>> allVertices;

		for (const auto& polygon : m_polygons) {
			bool inside = false;
			for (const auto& outline : polygon) {
				for (const auto& p : outline.container()) {
					for (auto curROI = 0; curROI < _ROIs.size() && !inside; curROI++) {
						inside = _ROIs[curROI]->inside(p.x(), p.y(), 0.f);
						if (inside) break;
					}
					if (inside) break;
				}
			}
			if (inside) {
				allVertices.push_back(std::vector < std::vector <poca::core::Vec3mf>>());
				auto& vertices = allVertices.back();
				for (const auto& outline : polygon) {
					vertices.push_back(std::vector<poca::core::Vec3mf>());
					for (const auto& p : outline.container()) {
						vertices.back().emplace_back(p.x(), p.y(), 0.f);
					}
				}
			}
		}
		return new ObjectListPolygon(allVertices);
	}

	poca::geometry::ObjectListPolygon* ObjectListPolygon::exportFilteredObjects()
	{
		std::vector <std::vector <std::vector <poca::core::Vec3mf>>> allVertices;

		for (auto n = 0; n < m_polygons.size(); n++) {
			if (m_selection[n]) {
				const auto& polygon = m_polygons[n];
				allVertices.push_back(std::vector < std::vector <poca::core::Vec3mf>>());
				auto& vertices = allVertices.back();
				for (const auto& outline : polygon) {
					vertices.push_back(std::vector<poca::core::Vec3mf>());
					for (const auto& p : outline.container()) {
						vertices.back().emplace_back(p.x(), p.y(), 0.f);
					}
				}
			}
		}
		return allVertices.empty() ? NULL: new ObjectListPolygon(allVertices);
	}
	
	void ObjectListPolygon::generateLocs(std::vector <poca::core::Vec3mf>& _locs)
	{
		/*_locs.clear();
		for (const auto& mesh : m_meshes)
			for (const auto& point : mesh.points())
				_locs.push_back(poca::core::Vec3mf(point.x(), point.y(), point.z()));*/
		_locs.resize(m_locs.nbData());
		const std::vector <uint32_t>& indices = m_locs.getData();
		for (size_t n = 0; n < indices.size(); n++) {
			size_t index = indices.at(n);
			_locs[n].set(m_xs[index], m_ys[index], 0.f);
		}
	}

	void ObjectListPolygon::generateNormalLocs(std::vector <poca::core::Vec3mf>& _norms)
	{
		_norms.resize(m_locs.nbData());
		const std::vector <uint32_t>& indices = m_locs.getData(), & objects = m_locs.getFirstElements();
		for (uint32_t n = 0; n < m_locs.nbElements(); n++) {
			uint32_t nbLocs = objects[n + 1] - objects[n];
			float nbD = nbLocs;
			poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
			for (uint32_t idx = objects[n]; idx < objects[n + 1]; idx++) {
				uint32_t index = indices[idx];
				centroid += poca::core::Vec3mf(m_xs[index], m_ys[index], 0.f) / nbD;
			}
			for (uint32_t idx = objects[n]; idx < objects[n + 1]; idx++) {
				uint32_t index = indices[idx];
				poca::core::Vec3mf normal = poca::core::Vec3mf(m_xs[index], m_ys[index], 0.f) - centroid;
				normal.normalize();
				_norms[idx] = normal;
			}
		}
	}

	void ObjectListPolygon::getLocsFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::getLocsFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		_features.resize(m_locs.nbData());// *2);

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::getOutlinesFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		_features.resize(m_outlines.nbData());// *2);

		size_t cpt = 0;
		for (size_t i = 0; i < m_outlines.nbElements(); i++) {
			for (size_t j = 0; j < m_outlines.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}
		}
	}

	void ObjectListPolygon::getOutlinesFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		_features.resize(m_outlines.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_outlines.nbElements(); i++) {
			for (size_t j = 0; j < m_outlines.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::generateLocsPickingIndices(std::vector <float>& _ids) const
	{
		/*_ids.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_ids.push_back(i + 1);
			i++;
		}*/
		_ids.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_ids[cpt++] = i + 1;
			}

		}
	}

	void ObjectListPolygon::generateTriangles(std::vector <poca::core::Vec3mf>& _triangles)
	{
		std::copy(m_triangles.getData().begin(), m_triangles.getData().end(), std::back_inserter(_triangles));
	}

	void ObjectListPolygon::generateOutlines(std::vector <poca::core::Vec3mf>& _outlines)
	{
		std::copy(m_outlines.getData().begin(), m_outlines.getData().end(), std::back_inserter(_outlines));
	}

	void ObjectListPolygon::generateNormals(std::vector <poca::core::Vec3mf>& _normals)
	{
		_normals.resize(m_triangles.getData().size());
		std::fill(_normals.begin(), _normals.end(), poca::core::Vec3mf(0.f, 0.f, 1.f));
	}

	void ObjectListPolygon::generatePickingIndices(std::vector <float>& _ids) const
	{
		const std::vector <poca::core::Vec3mf> triangles = m_triangles.getData();
		_ids.resize(triangles.size());

		size_t cpt = 0;
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j++) {
				_ids[cpt++] = i + 1;
			}

		}
	}

	void ObjectListPolygon::getFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		const std::vector <poca::core::Vec3mf> triangles = m_triangles.getData();
		_features.resize(triangles.size());

		size_t cpt = 0;
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::getFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		const std::vector <poca::core::Vec3mf> triangles = m_triangles.getData();
		_features.resize(triangles.size());

		size_t cpt = 0;
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	poca::core::BoundingBox ObjectListPolygon::computeBoundingBoxElement(const int _idx) const
	{
		return m_bboxMeshes[_idx];
	}

	poca::core::Vec3mf ObjectListPolygon::computeBarycenterElement(const int _idx) const
	{
		return m_centroids[_idx];
	}

	void ObjectListPolygon::generateOutlineLocs(std::vector <poca::core::Vec3mf>& _locs)
	{
		/*_locs.clear();
		for (const auto& mesh : m_meshes)
			for (const auto& point : mesh.points())
				_locs.push_back(poca::core::Vec3mf(point.x(), point.y(), point.z()));*/
		_locs.resize(m_locs.nbData());
		const std::vector <uint32_t>& indices = m_locs.getData();
		for (size_t n = 0; n < indices.size(); n++) {
			size_t index = indices.at(n);
			_locs[n].set(m_xs[index], m_ys[index], 0.f);
		}
	}

	void ObjectListPolygon::getOutlineLocsFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		/*_features.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_features.push_back(_selection[i] ? _values[i] : _notSelectedValue);
			i++;
		}*/
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::getOutlineLocsFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		/*_features.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_features.push_back(_selection[i] ? _selectedValue : _notSelectedValue);
			i++;
		}*/
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::saveAsPol(const std::string& _filename) const
	{
		std::ofstream fs(_filename, std::ifstream::binary);
		size_t nb = m_polygons.size();
		fs.write(reinterpret_cast<char*>(&nb), sizeof(size_t));
		for (const auto& polygons : m_polygons) {
			nb = polygons.size();
			fs.write(reinterpret_cast<char*>(&nb), sizeof(size_t));

			for (const auto& poly : polygons) {
				std::vector <poca::core::Vec3mf> vertices;
				for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it)
					vertices.emplace_back(it->x(), it->y(), 0.f);
				nb = vertices.size();
				fs.write(reinterpret_cast<char*>(&nb), sizeof(size_t));
				fs.write(reinterpret_cast<char*>(vertices.data()), nb * sizeof(poca::core::Vec3mf));
			}
		}
		fs.close();
	}
}
