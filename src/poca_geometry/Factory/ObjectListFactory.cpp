/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListFactory.cpp
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

#include <fstream>

#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/Alpha_shape_3.h>
#include <CGAL/Alpha_shape_cell_base_3.h>
#include <CGAL/Alpha_shape_vertex_base_3.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/boost/graph/copy_face_graph.h>
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
#include <CGAL/Point_set_3.h>
#include <CGAL/Poisson_reconstruction_function.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Kernel/global_functions_2.h>
#endif
#include <thrust/host_vector.h>
#include <thrust\sort.h>
#include <thrust\functional.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>

#include <QtWidgets/QMessageBox>

#include <General/BasicComponent.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/ROIInterface.hpp>
#include <General/Histogram.hpp>
#include <General/Engine.hpp>
#include <General/Misc.h>
#include <Geometry/BasicComputation.hpp>

#include "ObjectListFactory.hpp"
#include "../Interfaces/DelaunayTriangulationInterface.hpp"
#include "../Interfaces/DelaunayTriangulationFactoryInterface.hpp"
#include "../Geometry/DelaunayTriangulation.hpp"
#include "../Geometry/BasicComputation.hpp"
#include "../Geometry/ObjectListPolygon.hpp"
#include "../Geometry/ObjectListDelaunay.hpp"
#include "../Geometry/ObjectListMesh.hpp"
#include "../Geometry/delaunator.hpp"
#include "../Geometry/CGAL_includes.hpp"

typedef CGAL::Alpha_shape_vertex_base_3<K_inexact>               Alpha_Vb;
typedef CGAL::Alpha_shape_cell_base_3<K_inexact>                 Alpha_Fb;
typedef CGAL::Triangulation_data_structure_3<Alpha_Vb, Alpha_Fb>      Alpha_Tds;
typedef CGAL::Delaunay_triangulation_3<K_inexact, Alpha_Tds, CGAL::Fast_location>  Alpha_Delaunay;
typedef CGAL::Alpha_shape_3<Alpha_Delaunay>                    Alpha_shape_3;
typedef Alpha_shape_3::Alpha_iterator                    Alpha_iterator;
typedef Alpha_shape_3::NT                                Alpha_NT;
typedef Alpha_shape_3::Cell_handle                          Alpha_Cell_handle;
typedef Alpha_shape_3::Vertex_handle                        Alpha_Vertex_handle;
typedef Alpha_shape_3::Facet                             Alpha_Facet;
typedef CGAL::Arr_segment_traits_2<K_inexact> Traits_2;
typedef CGAL::Arrangement_2<Traits_2> Arrangement_2;

#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
typedef CGAL::Point_set_3<Point_3_inexact, Vector_3_inexact> Point_set;
#endif

/*// Each edge is a pair of vertex indices
typedef std::pair<size_t, size_t> EdgeObjectPolygon;

// Undirected edge comparison
EdgeObjectPolygon make_ordered_edge(size_t a, size_t b) {
	return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

// Adjacency list: from vertex index to connected vertex indices
std::unordered_map<size_t, std::vector<size_t>> build_boundary_graph(const std::vector<EdgeObjectPolygon>& boundary_edges) {
	std::unordered_map<size_t, std::vector<size_t>> adjacency;

	for (const auto& e : boundary_edges) {
		size_t a = e.first, b = e.second;
		adjacency[a].push_back(b);
		adjacency[b].push_back(a);
	}

	return adjacency;
}

std::vector<std::vector<size_t>> extract_loops(const std::unordered_map<size_t, std::vector<size_t>>& adjacency) {
	std::unordered_set<size_t> visited_vertices;
	std::vector<std::vector<size_t>> loops;

	std::cout << "***************************" << std::endl;
	for (const auto& [start, _] : adjacency) {
		if (visited_vertices.count(start)) continue;

		std::vector<size_t> loop;
		auto current = start;
		auto previous = -1;
		std::cout << "------------------------------" << std::endl;
		std::cout << start << std::endl;
		do {
			loop.push_back(current);
			visited_vertices.insert(current);

			const auto& neighbors = adjacency.at(current);
			size_t next = -1;

			// Select the neighbor that is not the previous node
			for (int neighbor : neighbors) {
				if (neighbor != previous) {
					next = neighbor;
					break;
				}
			}

			previous = current;
			current = next;

			std::cout << current << std::endl;
		} while (current != start && current != -1);
		std::cout << "------------------------------" << std::endl;

		if (!loop.empty() && loop.front() == loop.back()) {
			loop.pop_back();  // Ensure closed loop
		}

		loops.push_back(loop);
	}
	std::cout << "***************************" << std::endl;

	return loops;
}

void build_polygons(
	const std::vector<std::vector<size_t>>& loops,
	std::vector<Polygon_2>& _polygons,
	const float* _xs, const float* _ys)
{
	std::vector<Polygon_2> polygons;

	for (const auto& loop : loops) {
		Polygon_2 poly;
		for (auto idx : loop) {
			poly.push_back(Point_2(_xs[idx], _ys[idx]));
		}

		_polygons.push_back(poly);
	}
}

void reorder_polygons_by_area(std::vector<Polygon_2>& polygons) {
	if (polygons.empty()) return;

	// Lambda to compute absolute area
	auto abs_area = [](const Polygon_2& poly) {
		return std::abs(poly.area());
		};

	// Find the iterator to the polygon with the largest area
	auto outer_it = std::max_element(polygons.begin(), polygons.end(),
		[&](const Polygon_2& a, const Polygon_2& b) {
			return abs_area(a) < abs_area(b);
		});

	// Move the outer polygon to the front (if it's not already there)
	if (outer_it != polygons.begin()) {
		std::iter_swap(polygons.begin(), outer_it);
	}

	// Ensure correct orientation:
	if (!polygons[0].is_clockwise_oriented()) {
		polygons[0].reverse_orientation();
	}

	for (size_t i = 1; i < polygons.size(); ++i) {
		if (!polygons[i].is_clockwise_oriented()) {
			polygons[i].reverse_orientation();
		}
	}
}*/

/*std::vector<std::vector<size_t>> extract_loops(
	const std::unordered_map<size_t, std::vector<size_t>>& adjacency,
	const float* _xs, const float* _ys)
{
	std::unordered_set<std::pair<size_t, size_t>, boost::hash<std::pair<size_t, size_t>>> visited_edges;
	std::vector<std::vector<size_t>> loops;

	for (const auto& [start_vertex, neighbors] : adjacency) {
		for (auto neighbor : neighbors) {

			auto edge_key = std::minmax(start_vertex, neighbor);
			if (visited_edges.count(edge_key)) continue;

			std::vector<size_t> loop;
			loop.push_back(start_vertex);

			int prev = start_vertex;
			int current = neighbor;
			loop.push_back(current);
			visited_edges.insert(edge_key);

			while (current != start_vertex) {
				const auto& next_neighbors = adjacency.at(current);
				int next_vertex = -1;
				double best_turn = -std::numeric_limits<double>::infinity();

				for (int candidate : next_neighbors) {
					if (candidate == prev) continue;

					Point_2 pprev(_xs[prev], _ys[prev]), pcurrent(_xs[current], _ys[current]), pcandidate(_xs[candidate], _ys[candidate]);

					CGAL::Orientation orient = CGAL::orientation(
						pprev, pcurrent, pcandidate);

					double score = 0;
					if (orient == CGAL::LEFT_TURN) score = 1;
					else if (orient == CGAL::COLLINEAR) score = 0;
					else score = -1;

					if (score > best_turn) {
						best_turn = score;
						next_vertex = candidate;
					}
				}

				if (next_vertex == -1) break;  // Dead end, should not happen in a closed boundary.

				prev = current;
				current = next_vertex;
				loop.push_back(current);

				std::cout << current << std::endl;

				visited_edges.insert(std::minmax(prev, current));
			}

			// If the loop isn't closed, close it
			//if (loop.front() != loop.back()) {
			//	loop.push_back(loop.front());
			//}
			if (loop.front() == loop.back())
				loop.pop_back();

			loops.push_back(loop);
		}
	}

	return loops;
}*/

std::vector<std::vector<size_t>> extract_loops(
	const std::unordered_map<size_t, std::vector<size_t>>& adjacency,
	const float* _xs, const float* _ys)
{
	typedef std::pair<size_t, size_t> Edge;
	std::unordered_set<Edge, boost::hash<Edge>> visited_edges;
	std::vector<std::vector<size_t>> loops;

	for (const auto& [start_vertex, neighbors] : adjacency) {
		for (auto neighbor : neighbors) {

			auto edge_key = std::minmax(start_vertex, neighbor);
			if (visited_edges.count(edge_key)) continue;

			// Decide initial walk direction based on the first two points
			Point_2 p0(_xs[start_vertex], _ys[start_vertex]);
			Point_2 p1(_xs[neighbor], _ys[neighbor]);
			poca::geometry::WalkDirection walk_dir = poca::geometry::WalkDirection::CCW; // default

			// Find a third neighbor if possible to decide orientation
			if (neighbors.size() >= 2) {
				for (auto n2 : neighbors) {
					if (n2 == neighbor) continue;
					Point_2 p2(_xs[n2], _ys[n2]);
					auto orient = CGAL::orientation(p0, p1, p2);
					if (orient == CGAL::LEFT_TURN) walk_dir = poca::geometry::WalkDirection::CCW;
					else if (orient == CGAL::RIGHT_TURN) walk_dir = poca::geometry::WalkDirection::CW;
					break;
				}
			}

			std::vector<size_t> loop;
			loop.push_back(start_vertex);

			size_t prev = start_vertex;
			size_t current = neighbor;
			loop.push_back(current);
			visited_edges.insert(edge_key);

			while (current != start_vertex) {
				const auto& next_neighbors = adjacency.at(current);
				size_t next_vertex = -1;
				double best_score = walk_dir == poca::geometry::WalkDirection::CCW ?
					-std::numeric_limits<double>::infinity() :
					std::numeric_limits<double>::infinity();

				for (size_t candidate : next_neighbors) {
					if (candidate == prev) continue;

					Point_2 pprev(_xs[prev], _ys[prev]);
					Point_2 pcurrent(_xs[current], _ys[current]);
					Point_2 pcandidate(_xs[candidate], _ys[candidate]);

					K_inexact::Vector_2 v1 = pcurrent - pprev;
					K_inexact::Vector_2 v2 = pcandidate - pcurrent;

					double angle = poca::geometry::angle_between(v1, v2);

					if ((walk_dir == poca::geometry::WalkDirection::CCW && angle > best_score) ||
						(walk_dir == poca::geometry::WalkDirection::CW && angle < best_score)) {
						best_score = angle;
						next_vertex = candidate;
					}
				}

				if (next_vertex == size_t(-1)) break;  // Should not happen in a closed loop

				prev = current;
				current = next_vertex;
				loop.push_back(current);

				visited_edges.insert(std::minmax(prev, current));
			}

			if (loop.front() == loop.back())
				loop.pop_back(); // Loop is already closed by walking

			auto loopProcessed = poca::geometry::remove_duplicate_loops<size_t>(loop);

			/*std::cout << "******************" << std::endl;
			for (auto id : loopProcessed.first)
				std::cout << id << std::endl;

			std::cout << "******************" << std::endl;*/

			loops.push_back(loopProcessed.first);

			for (auto loopRemoved : loopProcessed.second) {
				bool loopExistinHoles = false;
				for (const auto& existingHole : loops) {
					loopExistinHoles = poca::geometry::have_same_elements<size_t>(loopRemoved, existingHole);
					if (loopExistinHoles)
						break;
				}
				if (!loopExistinHoles)
					loops.push_back(loopRemoved);
			}
		}
	}

	/*for (auto l : loops) {
		std::cout << "*********************************" << std::endl;
		std::copy(l.begin(), l.end(),
			std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;
		std::cout << "-----------------------------" << std::endl;
		std::sort(l.begin(), l.end());
		std::copy(l.begin(), l.end(),
			std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;
	}*/

	return loops;
}

void build_polygons(
	const std::vector<std::vector<size_t>>& loops,
	std::vector<Polygon_2>& _polygons,
	const float* _xs, const float* _ys)
{
	std::vector<Polygon_2> polygons;

	for (const auto& loop : loops) {
		Polygon_2 poly;
		//std::cout << "**************************************" << std::endl;
		for (auto idx : loop) {
			poly.push_back(Point_2(_xs[idx], _ys[idx]));
			//std::cout << idx << " - " << (idx == 8293) << std::endl;
		}
		//std::cout << std::endl;

		_polygons.push_back(poly);
	}
}

namespace poca::geometry {
	ObjectListFactoryInterface* createObjectListFactory()
	{
		return new ObjectListFactory();
	}

	ObjectListFactory::ObjectListFactory()
	{

	}

	ObjectListFactory::~ObjectListFactory()
	{

	}

	ObjectListInterface* ObjectListFactory::createObjectList(poca::core::MyObjectInterface* _obj, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const bool _inROIs)
	{
		poca::core::BasicComponentInterface* bci = _obj->getBasicComponent("DelaunayTriangulation");
		DelaunayTriangulationInterface* delaunay = dynamic_cast <DelaunayTriangulationInterface*>(bci);
		if (!delaunay) return NULL;
		const std::vector <poca::core::ROIInterface*>& ROIs = _inROIs ? _obj->getROIs() : std::vector <poca::core::ROIInterface*>();
		return createObjectList(delaunay, _selection, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea, ROIs);
	}
	ObjectListInterface* ObjectListFactory::createObjectList(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		std::vector <bool> selectionDelaunay;
		_delaunay->generateFaceSelectionFromLocSelection(_selection, selectionDelaunay);
		return createObjectListFromDelaunay(_delaunay, selectionDelaunay, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea, _ROIs);
	}

	ObjectListInterface* ObjectListFactory::createObjectListFromDelaunay(poca::core::MyObjectInterface* _obj, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const bool _inROIs)
	{
		poca::core::BasicComponentInterface* bci = _obj->getBasicComponent("DelaunayTriangulation");
		DelaunayTriangulationInterface* delaunay = dynamic_cast <DelaunayTriangulationInterface*>(bci);
		if (!delaunay) {
			QMessageBox msgBox;
			msgBox.setText("Delaunay triangulation is required to create the objects.");
			msgBox.exec();
			return NULL;
		}
		const std::vector <poca::core::ROIInterface*>& ROIs = _inROIs ? _obj->getROIs() : std::vector <poca::core::ROIInterface*>();
		return createObjectListFromDelaunay(delaunay, _selection, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea, ROIs);
	}

	ObjectListInterface* ObjectListFactory::createObjectListFromDelaunay(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		clock_t t1, t2;
		t1 = clock();
		ObjectListInterface* objs = NULL;
		if (_delaunay->dimension() == 2)
			objs = createObjectList2D(_delaunay, _selection, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea, _ROIs);
		else if (_delaunay->dimension() == 3)
			objs = createObjectList3D(_delaunay, _selection, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea, _ROIs);
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("Time for creating objects: %ld ms\n", elapsed);
		return objs;
	}

	ObjectListInterface* ObjectListFactory::createObjectListAlreadyIdentified(poca::core::MyObjectInterface* _obj, const std::vector <uint32_t>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea)
	{
		poca::core::BasicComponentInterface* bci = _obj->getBasicComponent("DelaunayTriangulation");
		DelaunayTriangulationInterface* delaunay = dynamic_cast <DelaunayTriangulationInterface*>(bci);
		if (!delaunay) {
			QMessageBox msgBox;
			msgBox.setText("Delaunay triangulation is required to create the objects.");
			msgBox.exec();
			return NULL;
		}
		return createObjectListAlreadyIdentified(delaunay, _selection, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea);
	}

	ObjectListInterface* ObjectListFactory::createObjectListAlreadyIdentified(DelaunayTriangulationInterface* _delaunay, const std::vector <uint32_t>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea)
	{
		clock_t t1, t2;
		t1 = clock();
		ObjectListInterface* objs = NULL;
		std::map <uint32_t, std::vector <uint32_t>> objects;
		for (auto n = 0; n < _selection.size(); n++) {
			auto index = _selection[n];
			if (index == std::numeric_limits<uint32_t>::max()) continue;
			if (objects.find(index) == objects.end())
				objects[index] = std::vector<uint32_t>();
			objects[index].push_back(n);
		}
		//std::map <uint32_t, std::vector <uint32_t>> selectionDelaunay;
		//_delaunay->generateFaceSelectionFromLocSelection(_selection, selectionDelaunay);
		if (_delaunay->dimension() == 2)
			objs = createObjectList2D(_delaunay, objects, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea);
		else if (_delaunay->dimension() == 3)
			objs = createObjectList3D(_delaunay, objects, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea);
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("Time for creating objects: %ld ms\n", elapsed);
		return objs;
	}

	ObjectListInterface* ObjectListFactory::createObjectList2D(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		std::vector <float> zsTmp;
		if (_delaunay->getZs() == NULL)
			zsTmp = std::vector<float>(_delaunay->nbPoints(), 0.f);
		const float* zs = _delaunay->getZs() == NULL ? zsTmp.data() : _delaunay->getZs();

		const std::vector<uint32_t>& triangles = _delaunay->getTriangles();
		const poca::core::MyArrayUInt32& neighbors = _delaunay->getNeighbors();

		std::vector <uint32_t> linkTriangulationFacesToObjects(_selection.size(), std::numeric_limits<std::uint32_t>::max());
		std::vector <bool> selectionTriangulationFaces(_selection);
		if (!_ROIs.empty()) {
			for (size_t n = 0; n < _delaunay->nbFaces(); n++) {
				if (!selectionTriangulationFaces[n]) continue;
				uint32_t i1 = triangles[3 * n], i2 = triangles[3 * n + 1], i3 = triangles[3 * n + 2];
				bool inside = false;
				for (size_t i = 0; i < _ROIs.size() && !inside; i++) {
					bool p1Inside = _ROIs[i]->inside(xs[i1], ys[i1], zs[i1]);
					bool p2Inside = _ROIs[i]->inside(xs[i2], ys[i2], zs[i2]);
					bool p3Inside = _ROIs[i]->inside(xs[i3], ys[i3], zs[i3]);
					inside = p1Inside && p2Inside && p3Inside;
				}
				selectionTriangulationFaces[n] = inside;
			}
		}

		bool applyCutDistance = _dMax != std::numeric_limits < double >::max();
		double dMaxSqr = applyCutDistance ? _dMax * _dMax : _dMax;
		for (size_t n = 0; n < _delaunay->nbFaces(); n++) {
			if (!selectionTriangulationFaces[n] || !applyCutDistance) continue;
			uint32_t i1 = triangles[3 * n], i2 = triangles[3 * n + 1], i3 = triangles[3 * n + 2];
			float d0 = distanceSqr(xs[i1], ys[i1], zs[i1], xs[i2], ys[i2], zs[i2]);
			float d1 = distanceSqr(xs[i2], ys[i2], zs[i2], xs[i3], ys[i3], zs[i3]);
			float d2 = distanceSqr(xs[i3], ys[i3], zs[i3], xs[i1], ys[i1], zs[i1]);
			selectionTriangulationFaces[n] = !(d0 > dMaxSqr || d1 > dMaxSqr || d2 > dMaxSqr);
		}

		std::vector <bool> originalSelection(selectionTriangulationFaces), selectionLocsForOutline(_delaunay->nbPoints(), false);

		std::vector <std::vector<Polygon_2>> polygons;
		std::vector <uint32_t> locsAllObjects, firstsLocs;
		uint32_t currentFirstLocs = 0, currentFirstTriangles = 0, currentFirstOutlines = 0;
		firstsLocs.push_back(currentFirstLocs);
		for (uint32_t n = 0; n < _delaunay->nbFaces(); n++) {
			if (!selectionTriangulationFaces[n]) continue;
			std::vector <uint32_t> queueTriangles;
			std::set <uint32_t> locsOfObject;
			queueTriangles.push_back(n);
			size_t currentTriangle = 0, sizeQueue = queueTriangles.size();

			std::vector<std::pair<size_t, size_t>> boundary_edges;

			float area = 0.f;
			while (currentTriangle < sizeQueue) {
				size_t index = queueTriangles.at(currentTriangle);
				if (selectionTriangulationFaces[index]) {
					selectionTriangulationFaces[index] = false;
					uint32_t i1 = triangles[3 * index], i3 = triangles[3 * index + 1], i2 = triangles[3 * index + 2];
					locsOfObject.insert(i1);
					locsOfObject.insert(i2);
					locsOfObject.insert(i3);
					poca::core::Vec3mf v1(xs[i1], ys[i1], zs[i1]), v2(xs[i2], ys[i2], zs[i2]), v3(xs[i3], ys[i3], zs[i3]);
					float sideA = (v1 - v2).length(), sideB = (v1 - v3).length(), sideC = (v2 - v3).length();
					area += poca::geometry::computeAreaTriangle<float>(sideA, sideB, sideC);

					for (uint32_t i = 0; i < neighbors.nbElementsObject(index); i++) {
						uint32_t indexNeigh = neighbors.elementIObject(index, i);
						if (indexNeigh != std::numeric_limits<std::uint32_t>::max() && selectionTriangulationFaces[indexNeigh])
							queueTriangles.push_back(indexNeigh);
						if (indexNeigh == std::numeric_limits<std::uint32_t>::max() || !originalSelection[indexNeigh]) {
							std::array<size_t, 3> edge = _delaunay->getOutline(index, i);
							boundary_edges.push_back(std::make_pair(edge[0], edge[1]));
						}
					}
					sizeQueue = queueTriangles.size();
				}
				currentTriangle++;
			}

			
			if (_minNbLocs <= locsOfObject.size() && locsOfObject.size() <= _maxNbLocs && _minArea <= area && area <= _maxArea) {
				size_t curObject = firstsLocs.size() - 1;
				for (const uint32_t val : queueTriangles)
					linkTriangulationFacesToObjects[val] = curObject;
				currentFirstLocs += locsOfObject.size();
				firstsLocs.push_back(currentFirstLocs);
				std::copy(locsOfObject.begin(), locsOfObject.end(), std::back_inserter(locsAllObjects));
				polygons.push_back(std::vector <Polygon_2>());
				std::cout << __LINE__ << std::endl;
				auto adjacency = poca::geometry::build_boundary_graph<size_t>(boundary_edges);
				std::cout << __LINE__ << std::endl;
				auto loops = extract_loops(adjacency, xs, ys);
				std::cout << __LINE__ << std::endl;
				build_polygons(loops, polygons.back(), xs, ys);
				std::cout << __LINE__ << std::endl;
				poca::geometry::reorder_polygons_by_area(polygons.back());
				std::cout << __LINE__ << std::endl;
			}
		}
		return locsAllObjects.empty() ? NULL : new ObjectListPolygon(xs, ys, _delaunay->getZs() == NULL ? NULL : _delaunay->getZs(), polygons, locsAllObjects, firstsLocs, linkTriangulationFacesToObjects);
	}

	ObjectListInterface* ObjectListFactory::createObjectList2D_old(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		std::vector <float> zsTmp;
		if (_delaunay->getZs() == NULL)
			zsTmp = std::vector<float>(_delaunay->nbPoints(), 0.f);
		const float* zs = _delaunay->getZs() == NULL ? zsTmp.data() : _delaunay->getZs();

		const std::vector<uint32_t>& triangles = _delaunay->getTriangles();
		const poca::core::MyArrayUInt32& neighbors = _delaunay->getNeighbors();

		std::vector <uint32_t> linkTriangulationFacesToObjects(_selection.size(), std::numeric_limits<std::uint32_t>::max());
		std::vector <bool> selectionTriangulationFaces(_selection);
		if (!_ROIs.empty()) {
			for (size_t n = 0; n < _delaunay->nbFaces(); n++) {
				if (!selectionTriangulationFaces[n]) continue;
				uint32_t i1 = triangles[3 * n], i2 = triangles[3 * n + 1], i3 = triangles[3 * n + 2];
				bool inside = false;
				for (size_t i = 0; i < _ROIs.size() && !inside; i++) {
					bool p1Inside = _ROIs[i]->inside(xs[i1], ys[i1], zs[i1]);
					bool p2Inside = _ROIs[i]->inside(xs[i2], ys[i2], zs[i2]);
					bool p3Inside = _ROIs[i]->inside(xs[i3], ys[i3], zs[i3]);
					inside = p1Inside && p2Inside && p3Inside;
				}
				selectionTriangulationFaces[n] = inside;
			}
		}

		bool applyCutDistance = _dMax != std::numeric_limits < double >::max();
		double dMaxSqr = applyCutDistance ? _dMax * _dMax : _dMax;
		for (size_t n = 0; n < _delaunay->nbFaces(); n++) {
			if (!selectionTriangulationFaces[n] || !applyCutDistance) continue;
			uint32_t i1 = triangles[3 * n], i2 = triangles[3 * n + 1], i3 = triangles[3 * n + 2];
			float d0 = distanceSqr(xs[i1], ys[i1], zs[i1], xs[i2], ys[i2], zs[i2]);
			float d1 = distanceSqr(xs[i2], ys[i2], zs[i2], xs[i3], ys[i3], zs[i3]);
			float d2 = distanceSqr(xs[i3], ys[i3], zs[i3], xs[i1], ys[i1], zs[i1]);
			selectionTriangulationFaces[n] = !(d0 > dMaxSqr || d1 > dMaxSqr || d2 > dMaxSqr);
		}

		std::vector <bool> originalSelection(selectionTriangulationFaces), selectionLocsForOutline(_delaunay->nbPoints(), false);

		std::vector <uint32_t> locsAllObjects, firstsLocs, firstTriangles, firstOutlines;
		std::vector <poca::core::Vec3mf> trianglesAllObjects, outlinesAllObjects;
		uint32_t currentFirstLocs = 0, currentFirstTriangles = 0, currentFirstOutlines = 0;
		firstsLocs.push_back(currentFirstLocs);
		firstTriangles.push_back(currentFirstTriangles);
		firstOutlines.push_back(currentFirstOutlines);
		for (uint32_t n = 0; n < _delaunay->nbFaces(); n++) {
			if (!selectionTriangulationFaces[n]) continue;
			std::vector <uint32_t> queueTriangles;
			std::set <uint32_t> locsOfObject;
			std::vector <poca::core::Vec3mf> trianglesOfObject, outlineOfObject;
			queueTriangles.push_back(n);
			size_t currentTriangle = 0, sizeQueue = queueTriangles.size();

			std::vector<std::pair<size_t, size_t>> boundary_edges;

			float area = 0.f;
			while (currentTriangle < sizeQueue) {
				size_t index = queueTriangles.at(currentTriangle);
				if (selectionTriangulationFaces[index]) {
					selectionTriangulationFaces[index] = false;
					uint32_t i1 = triangles[3 * index], i3 = triangles[3 * index + 1], i2 = triangles[3 * index + 2];
					locsOfObject.insert(i1);
					locsOfObject.insert(i2);
					locsOfObject.insert(i3);
					poca::core::Vec3mf v1(xs[i1], ys[i1], zs[i1]), v2(xs[i2], ys[i2], zs[i2]), v3(xs[i3], ys[i3], zs[i3]);
					trianglesOfObject.push_back(v1);
					trianglesOfObject.push_back(v2);
					trianglesOfObject.push_back(v3);
					float sideA = (v1 - v2).length(), sideB = (v1 - v3).length(), sideC = (v2 - v3).length();
					area += poca::geometry::computeAreaTriangle<float>(sideA, sideB, sideC);

					for (uint32_t i = 0; i < neighbors.nbElementsObject(index); i++) {
						uint32_t indexNeigh = neighbors.elementIObject(index, i);
						if (indexNeigh != std::numeric_limits<std::uint32_t>::max() && selectionTriangulationFaces[indexNeigh])
							queueTriangles.push_back(indexNeigh);
						if (indexNeigh == std::numeric_limits<std::uint32_t>::max() || !originalSelection[indexNeigh]) {
							std::array<size_t, 3> edge = _delaunay->getOutline(index, i);
							//outlineOfObject.push_back(poca::core::Vec3mf(xs[edge[0]], ys[edge[0]], zs[edge[0]]));
							//outlineOfObject.push_back(poca::core::Vec3mf(xs[edge[1]], ys[edge[1]], zs[edge[1]]));
							boundary_edges.push_back(std::make_pair(edge[0], edge[1]));
						}
					}
					sizeQueue = queueTriangles.size();
				}
				currentTriangle++;
			}

			if (_minNbLocs <= locsOfObject.size() && locsOfObject.size() <= _maxNbLocs && _minArea <= area && area <= _maxArea) {
				std::vector<Polygon_2> polygons;
				auto adjacency = build_boundary_graph(boundary_edges);
				auto loops = extract_loops(adjacency, xs, ys);
				build_polygons(loops, polygons, xs, ys);
				for (const auto& polygon : polygons) {
					const auto& points = polygon.container();
					std::size_t n = points.size();

					for (std::size_t i = 0; i < n; ++i) {
						const auto& curr = points[i];
						const auto& next = points[(i + 1) % n];  // wrap around

						outlineOfObject.emplace_back(curr.x(), curr.y(), 0.f);
						outlineOfObject.emplace_back(next.x(), next.y(), 0.f);
					}
				}

				size_t curObject = firstsLocs.size() - 1;
				for (const uint32_t val : queueTriangles)
					linkTriangulationFacesToObjects[val] = curObject;
				currentFirstLocs += locsOfObject.size();
				currentFirstTriangles += trianglesOfObject.size();
				currentFirstOutlines += outlineOfObject.size();
				firstsLocs.push_back(currentFirstLocs);
				firstTriangles.push_back(currentFirstTriangles);
				firstOutlines.push_back(currentFirstOutlines);
				std::copy(locsOfObject.begin(), locsOfObject.end(), std::back_inserter(locsAllObjects));
				std::copy(trianglesOfObject.begin(), trianglesOfObject.end(), std::back_inserter(trianglesAllObjects));
				std::copy(outlineOfObject.begin(), outlineOfObject.end(), std::back_inserter(outlinesAllObjects));
			}
		}
		return locsAllObjects.empty() ? NULL : new ObjectListDelaunay(xs, ys, _delaunay->getZs() == NULL ? NULL : _delaunay->getZs(), locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, outlinesAllObjects, firstOutlines, linkTriangulationFacesToObjects);
	}

	ObjectListInterface* ObjectListFactory::createObjectList3D(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		const float* zs = _delaunay->getZs();
		const std::vector <float>& volumes = static_cast<poca::core::Histogram<float>*>(_delaunay->getOriginalHistogram("volume"))->getValues();
		const std::vector<uint32_t>& triangles = _delaunay->getTriangles();
		const poca::core::MyArrayUInt32& neighbors = _delaunay->getNeighbors();
		const std::vector <uint32_t> indiceTriangles = neighbors.getFirstElements();

		std::vector <bool> originalSelectionTriangulationFaces(_selection);
		uint32_t debugNbSelect = 0;
		if (!_ROIs.empty()) {
			for (size_t n = 0; n < _delaunay->nbFaces(); n++) {
				if (!originalSelectionTriangulationFaces[n]) continue;
				uint32_t index = indiceTriangles[n];
				uint32_t i1 = triangles[3 * index],
					i2 = triangles[3 * index + 3 * 1],
					i3 = triangles[3 * index + 3 * 2],
					i4 = triangles[3 * index + 3 * 3];
				bool inside = false;
				for (size_t i = 0; i < _ROIs.size() && !inside; i++) {
					bool p1Inside = _ROIs[i]->inside(xs[i1], ys[i1], zs[i1]);
					bool p2Inside = _ROIs[i]->inside(xs[i2], ys[i2], zs[i2]);
					bool p3Inside = _ROIs[i]->inside(xs[i3], ys[i3], zs[i3]);
					bool p4Inside = _ROIs[i]->inside(xs[i4], ys[i4], zs[i4]);
					inside = p1Inside && p2Inside && p3Inside && p4Inside;
				}
				originalSelectionTriangulationFaces[n] = inside;
				if (originalSelectionTriangulationFaces[n]) debugNbSelect++;
			}
		}

		bool applyCutDistance = _dMax != std::numeric_limits < float >::max();
		double dMaxSqr = applyCutDistance ? _dMax * _dMax : _dMax;
		if (applyCutDistance) {
			for (uint32_t n = 0; n < _delaunay->nbFaces(); n++) {
				if (!originalSelectionTriangulationFaces[n] || !applyCutDistance) continue;
				for (uint32_t i = indiceTriangles[n]; i < indiceTriangles[n + 1] && originalSelectionTriangulationFaces[n]; i++) {
					uint32_t i1 = triangles[3 * i], i2 = triangles[3 * i + 1], i3 = triangles[3 * i + 2];
					float d0 = distanceSqr(xs[i1], ys[i1], zs[i1], xs[i2], ys[i2], zs[i2]);
					float d1 = distanceSqr(xs[i2], ys[i2], zs[i2], xs[i3], ys[i3], zs[i3]);
					float d2 = distanceSqr(xs[i3], ys[i3], zs[i3], xs[i1], ys[i1], zs[i1]);
					originalSelectionTriangulationFaces[n] = !(d0 > dMaxSqr || d1 > dMaxSqr || d2 > dMaxSqr);
				}
			}
		}
	
		std::vector <uint32_t> linkTriangulationFacesToObjects(_selection.size(), std::numeric_limits<std::uint32_t>::max());
		std::vector <bool> selectionTriangulationFaces(originalSelectionTriangulationFaces);

		std::vector <uint32_t> locsAllObjects, firstsLocs, firstTriangles, locsAllOutlines, firstOutlineLocs;
		std::vector <poca::core::Vec3mf> trianglesAllObjects, normalsAllOutlineLocs;
		uint32_t currentFirstLocs = 0, currentFirstTriangles = 0, currentFirstOutlineLocs = 0;
		firstsLocs.push_back(currentFirstLocs);
		firstTriangles.push_back(currentFirstTriangles);
		firstOutlineLocs.push_back(currentFirstOutlineLocs);
		std::vector <float> volumeObjects;
		float volume = 0.f;
		double volumeD = 0.;
		std::vector <uint32_t> allIndexesTriangles;

		//For ObjectListMesh
		std::vector <std::vector <poca::core::Vec3mf>> meshPoints;
		std::vector <std::vector <std::vector <std::size_t>>> meshTris;
		std::vector < Surface_mesh_3_double> meshes;

		for (uint32_t n = 0; n < _delaunay->nbFaces(); n++) {
			if (!selectionTriangulationFaces[n]) continue;
			volume = 0.f;
			volumeD = 0.;
			std::vector <uint32_t> queueTriangles, indexTrianglesOfObject;
			std::set <uint32_t> locsOfObject, locsOfOutline;
			std::vector <poca::core::Vec3mf> trianglesOfObject, normalsTrianglesOfObject, normalOutlineLocObject;
			queueTriangles.push_back(n);
			selectionTriangulationFaces[n] = false;
			uint32_t currentTriangle = 0, sizeQueue = queueTriangles.size();
			while (currentTriangle < sizeQueue) {
				uint32_t indexFace = queueTriangles.at(currentTriangle);
				{
					
					uint32_t index = indiceTriangles[indexFace];
					//Here we have a tetrahedron that is composed of 4 triangles -> 12 vertices
					//If we want to find the 4 vertices, we have to use the first vertex of the four triangles
					//Then we need to determine if a traingle is at the border of the object to add it
					uint32_t is[4] = { triangles[3 * index],
						triangles[3 * index + 3 * 1],
						triangles[3 * index + 3 * 2],
						triangles[3 * index + 3 * 3] };
					for(uint32_t ind : is)
						locsOfObject.insert(ind);
					
					volume += volumes[indexFace];
					volumeD += volumes[indexFace];

					for (uint32_t i = 0; i < neighbors.nbElementsObject(indexFace); i++) {
						uint32_t indexNeigh = neighbors.elementIObject(indexFace, i);
						
						if (indexNeigh == std::numeric_limits<std::uint32_t>::max()) {
							poca::core::Vec3mf centroidN;
							for (uint32_t idTmp : is)
								centroidN += (poca::core::Vec3mf(xs[idTmp], ys[idTmp], zs[idTmp]) / 4.f);
							uint32_t ids[3] = { is[(i + 1) % 4] , is[(i + 2) % 4], is[(i + 3) % 4] };
							poca::core::Vec3mf vs[3] = { poca::core::Vec3mf(xs[ids[0]], ys[ids[0]], zs[ids[0]]), poca::core::Vec3mf(xs[ids[1]], ys[ids[1]], zs[ids[1]]), poca::core::Vec3mf(xs[ids[2]], ys[ids[2]], zs[ids[2]]) }, centroidF(0, 0, 0);
							poca::core::Vec3mf e1 = vs[1] - vs[0], e2 = vs[2] - vs[0], e3 = centroidN - vs[0], normal = e1.cross(e2);
							normal.normalize();
							e3.normalize();
							if (normal.dot(e3) < 0.f)
								std::reverse(std::begin(ids), std::end(ids));
							else
								normal = -normal;
							for (const uint32_t id : ids) {
								indexTrianglesOfObject.push_back(id);
								locsOfOutline.insert(id);
								centroidF += poca::core::Vec3mf(xs[id], ys[id], zs[id]) / 3.f;
							}
							normalsTrianglesOfObject.push_back(normal);
						}
						else if (!originalSelectionTriangulationFaces[indexNeigh]) {
							uint32_t indexN = indiceTriangles[indexNeigh];
							uint32_t isN[4] = { triangles[3 * indexN],
								triangles[3 * indexN + 3 * 1],
								triangles[3 * indexN + 3 * 2],
								triangles[3 * indexN + 3 * 3] };
							std::vector <uint32_t> indexCurAndNeighLocs = { is[0], is[1], is[2], is[3], isN[0], isN[1], isN[2], isN[3] };
							std::map <uint32_t, int> duplicates;
							poca::core::findDuplicates(indexCurAndNeighLocs, duplicates);
							if (duplicates.size() != 3)
								std::cout << "Seems to have a problem" << std::endl;
							else {
								//try to keep the same orientation (cw or ccw) for all triangles
								//To achieve that, we compute the normal of the triangle
								//and compare to the vector coming from one point of the triangle to the centroid of the tetrahedron
								//if their dot product is positive, they are having the same orientation: we change the order of the triangle vertices
								//if it's negative, no change is needed
								poca::core::Vec3mf centroidN;
								for (uint32_t idTmp : is)
									centroidN += (poca::core::Vec3mf(xs[idTmp], ys[idTmp], zs[idTmp]) / 4.f);
								uint32_t ids[3];
								poca::core::Vec3mf vs[3], centroidF(0, 0, 0);
								size_t cptt = 0;
								for (std::map <uint32_t, int>::const_iterator it = duplicates.begin(); it != duplicates.end(); it++, cptt++) {
									vs[cptt].set(xs[it->first], ys[it->first], zs[it->first]);
									ids[cptt] = it->first;
								}
								poca::core::Vec3mf e1 = vs[1] - vs[0], e2 = vs[2] - vs[0], e3 = centroidN - vs[0], normal = e1.cross(e2);
								normal.normalize();
								e3.normalize();
								if (normal.dot(e3) < 0.f)
									std::reverse(std::begin(ids), std::end(ids));
								else
									normal = -normal;
								for (const uint32_t id : ids) {
									indexTrianglesOfObject.push_back(id);
									locsOfOutline.insert(id);
									centroidF += poca::core::Vec3mf(xs[id], ys[id], zs[id]) / 3.f;
								}
								normalsTrianglesOfObject.push_back(normal);
							}
						}
						else if (selectionTriangulationFaces[indexNeigh]) {
							queueTriangles.push_back(indexNeigh);
							selectionTriangulationFaces[indexNeigh] = false;
						}
					}
					sizeQueue = queueTriangles.size();
				}
				currentTriangle++;
			}

			if (_minNbLocs <= locsOfObject.size() && locsOfObject.size() <= _maxNbLocs && _minArea <= volume && volume <= _maxArea) {
				ObjectListFactoryInterface::TypeShape type = poca::core::Engine::instance()->getGlobalParameters()["typeObject"].get<ObjectListFactoryInterface::TypeShape>();
				switch (type) {
				case ObjectListFactoryInterface::TRIANGULATION:
					for (const auto id : indexTrianglesOfObject)
						trianglesOfObject.push_back(poca::core::Vec3mf(xs[id], ys[id], zs[id]));
					break;
				case ObjectListFactoryInterface::CONVEX_HULL:
					computeConvexHullObject3DFromOutline(xs, ys, zs, locsOfOutline, trianglesOfObject, volume);
					break;
				case ObjectListFactoryInterface::POISSON_SURFACE:
					//computePoissonSurfaceObject(xs, ys, zs, locsOfOutline, indexTrianglesOfObject, normalsTrianglesOfObject, trianglesOfObject, volume);
					computePoissonSurfaceObjectOMesh(xs, ys, zs, locsOfOutline, indexTrianglesOfObject, normalsTrianglesOfObject, volume, meshes);
					break;
				case ObjectListFactoryInterface::ALPHA_SHAPE:
					computeAlphaShape(xs, ys, zs, locsOfOutline, trianglesOfObject, volume);
					break;
				default:
					break;
				}

				size_t curObject = firstsLocs.size() - 1;
				for (const uint32_t val : queueTriangles) {
					linkTriangulationFacesToObjects[val] = curObject;
				}
				currentFirstLocs += locsOfObject.size();
				firstsLocs.push_back(currentFirstLocs);
				std::copy(locsOfObject.begin(), locsOfObject.end(), std::back_inserter(locsAllObjects));

				currentFirstOutlineLocs += locsOfOutline.size();
				firstOutlineLocs.push_back(currentFirstOutlineLocs);
				std::copy(locsOfOutline.begin(), locsOfOutline.end(), std::back_inserter(locsAllOutlines));

				currentFirstTriangles += trianglesOfObject.size();
				firstTriangles.push_back(currentFirstTriangles);
				std::copy(trianglesOfObject.begin(), trianglesOfObject.end(), std::back_inserter(trianglesAllObjects));

				computeNormalOfLocsObject(locsOfOutline, indexTrianglesOfObject, normalsTrianglesOfObject, normalOutlineLocObject);
				std::copy(normalOutlineLocObject.begin(), normalOutlineLocObject.end(), std::back_inserter(normalsAllOutlineLocs));

				volumeObjects.push_back(volume);

				std::copy(indexTrianglesOfObject.begin(), indexTrianglesOfObject.end(), std::back_inserter(allIndexesTriangles));

				//For ObjectListMesh
				thrust::host_vector <uint32_t> h_triangles(indexTrianglesOfObject), h_data(indexTrianglesOfObject);
				//std::copy(h_data.begin(), h_data.end(), std::ostream_iterator<uint32_t>(std::cout, " "));
				//std::cout << std::endl;
				thrust::sort(thrust::host, h_triangles.begin(), h_triangles.end());
				thrust::host_vector<uint32_t> h_unique = h_triangles;
				const auto end = thrust::unique(h_unique.begin(), h_unique.end());
				auto nbVertices = thrust::distance(h_unique.begin(), end);

				meshPoints.push_back(std::vector <poca::core::Vec3mf>());
				meshPoints.back().resize(nbVertices);
				for (auto i = 0; i < nbVertices; i++) {
					meshPoints.back()[i].set(xs[h_unique[i]], ys[h_unique[i]], zs[h_unique[i]]);
				}

				//relabel the faces
				thrust::lower_bound(h_unique.begin(), end, h_data.begin(), h_data.end(), h_triangles.begin());
				meshTris.push_back(std::vector <std::vector <std::size_t>>());
				for (auto i = 0; i < h_triangles.size(); i += 3){
					meshTris.back().push_back(std::vector<std::size_t>{h_triangles[i + 2], h_triangles[i + 1], h_triangles[i]});
				}
			}
		}
		//ObjectListInterface* objs = locsAllObjects.empty() ? NULL : new ObjectListDelaunay(xs, ys, zs, locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, volumeObjects, linkTriangulationFacesToObjects, locsAllOutlines, firstOutlineLocs, normalsAllOutlineLocs);
		
		ObjectListInterface* objs;
		ObjectListFactoryInterface::TypeShape type = poca::core::Engine::instance()->getGlobalParameters()["typeObject"].get<ObjectListFactoryInterface::TypeShape>();
		if (type == ObjectListFactoryInterface::POISSON_SURFACE) {
			objs = locsAllObjects.empty() ? NULL : new poca::geometry::ObjectListMesh(meshes);
		}
		else if (type == ObjectListFactoryInterface::TRIANGULATION) {
			objs = locsAllObjects.empty() ? NULL : new ObjectListDelaunay(xs, ys, zs, locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, volumeObjects, linkTriangulationFacesToObjects, locsAllOutlines, firstOutlineLocs, normalsAllOutlineLocs);
		}
		else {
			double targetLength = poca::core::Engine::instance()->getGlobalParameters()["meshTargetLength"].get<double>();
			int iterations = poca::core::Engine::instance()->getGlobalParameters()["meshIterations"].get<int>();

			std::vector <poca::core::ROIInterface*> ROIs;
			objs = locsAllObjects.empty() ? NULL : new poca::geometry::ObjectListMesh(meshPoints, meshTris, ROIs, true, true, targetLength, iterations);
		}

		return objs;
	}

	/*ObjectList* ObjectListFactory::createObjectList2D(DelaunayTriangulationInterface* _delaunay, const std::vector <uint32_t>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		std::vector <float> zsTmp;
		if (_delaunay->getZs() == NULL)
			zsTmp = std::vector<float>(_delaunay->nbPoints(), 0.f);
		const float* zs = _delaunay->getZs() == NULL ? zsTmp.data() : _delaunay->getZs();

		const std::vector<uint32_t>& triangles = _delaunay->getTriangles();
		const poca::core::MyArrayUInt32& neighbors = _delaunay->getNeighbors();

		std::vector <uint32_t> linkTriangulationFacesToObjects(_selection.size(), std::numeric_limits<std::uint32_t>::max());
		std::vector <uint32_t> selectionTriangulationFaces(_selection);
		if (!_ROIs.empty()) {
			for (size_t n = 0; n < _delaunay->nbFaces(); n++) {
				if (selectionTriangulationFaces[n] == std::numeric_limits<uint32_t>::max()) continue;
				uint32_t i1 = triangles[3 * n], i2 = triangles[3 * n + 1], i3 = triangles[3 * n + 2];
				bool inside = false;
				for (size_t i = 0; i < _ROIs.size() && !inside; i++) {
					bool p1Inside = _ROIs[i]->inside(xs[i1], ys[i1], zs[i1]);
					bool p2Inside = _ROIs[i]->inside(xs[i2], ys[i2], zs[i2]);
					bool p3Inside = _ROIs[i]->inside(xs[i3], ys[i3], zs[i3]);
					inside = p1Inside && p2Inside && p3Inside;
				}
				selectionTriangulationFaces[n] = inside;
			}
		}

		bool applyCutDistance = _dMax != std::numeric_limits < double >::max();
		double dMaxSqr = applyCutDistance ? _dMax * _dMax : _dMax;
		for (size_t n = 0; n < _delaunay->nbFaces(); n++) {
			if (selectionTriangulationFaces[n] == std::numeric_limits<uint32_t>::max() || !applyCutDistance) continue;
			uint32_t i1 = triangles[3 * n], i2 = triangles[3 * n + 1], i3 = triangles[3 * n + 2];
			float d0 = distanceSqr(xs[i1], ys[i1], zs[i1], xs[i2], ys[i2], zs[i2]);
			float d1 = distanceSqr(xs[i2], ys[i2], zs[i2], xs[i3], ys[i3], zs[i3]);
			float d2 = distanceSqr(xs[i3], ys[i3], zs[i3], xs[i1], ys[i1], zs[i1]);
			if (d0 > dMaxSqr || d1 > dMaxSqr || d2 > dMaxSqr)
				selectionTriangulationFaces[n] = std::numeric_limits<uint32_t>::max();
		}

		std::vector <uint32_t> originalSelection(selectionTriangulationFaces);

		std::vector <uint32_t> locsAllObjects, firstsLocs, firstTriangles, firstOutlines;
		std::vector <poca::core::Vec3mf> trianglesAllObjects, outlinesAllObjects;
		uint32_t currentFirstLocs = 0, currentFirstTriangles = 0, currentFirstOutlines = 0;
		firstsLocs.push_back(currentFirstLocs);
		firstTriangles.push_back(currentFirstTriangles);
		firstOutlines.push_back(currentFirstOutlines);
		float area = 0.f;
		for (uint32_t n = 0; n < _delaunay->nbFaces(); n++) {
			if (selectionTriangulationFaces[n] != std::numeric_limits<uint32_t>::max()) continue;
			uint32_t indexFoundTriangle = selectionTriangulationFaces[n];
			std::vector <uint32_t> queueTriangles;
			std::set <uint32_t> locsOfObject;
			std::vector <poca::core::Vec3mf> trianglesOfObject, outlineOfObject;
			queueTriangles.push_back(n);
			size_t currentTriangle = 0, sizeQueue = queueTriangles.size();
			while (currentTriangle < sizeQueue) {
				size_t index = queueTriangles.at(currentTriangle);
				if (selectionTriangulationFaces[index] != std::numeric_limits<uint32_t>::max()) {
					selectionTriangulationFaces[index] = std::numeric_limits<uint32_t>::max();
					uint32_t i1 = triangles[3 * index], i2 = triangles[3 * index + 1], i3 = triangles[3 * index + 2];
					locsOfObject.insert(i1);
					locsOfObject.insert(i2);
					locsOfObject.insert(i3);
					poca::core::Vec3mf v1(xs[i1], ys[i1], zs[i1]), v2(xs[i2], ys[i2], zs[i2]), v3(xs[i3], ys[i3], zs[i3]);
					trianglesOfObject.push_back(v1);
					trianglesOfObject.push_back(v2);
					trianglesOfObject.push_back(v3);
					float sideA = (v1 - v2).length(), sideB = (v1 - v3).length(), sideC = (v2 - v3).length();
					area += poca::geometry::computeAreaTriangle<float>(sideA, sideB, sideC);

					for (uint32_t i = 0; i < neighbors.nbElementsObject(index); i++) {
						uint32_t indexNeigh = neighbors.elementIObject(index, i);
						if (indexNeigh != std::numeric_limits<std::uint32_t>::max() && selectionTriangulationFaces[indexNeigh])
							queueTriangles.push_back(indexNeigh);
						if (indexNeigh == std::numeric_limits<std::uint32_t>::max() || originalSelection[indexNeigh] == std::numeric_limits<std::uint32_t>::max()) {
							std::array<size_t, 3> edge = _delaunay->getOutline(index, i);
							outlineOfObject.push_back(poca::core::Vec3mf(xs[edge[0]], ys[edge[0]], zs[edge[0]]));
							outlineOfObject.push_back(poca::core::Vec3mf(xs[edge[1]], ys[edge[1]], zs[edge[1]]));
						}
					}
					sizeQueue = queueTriangles.size();
				}
				currentTriangle++;
			}
			if (_minNbLocs <= locsOfObject.size() && locsOfObject.size() <= _maxNbLocs && _minArea <= area && area <= _maxArea) {
				size_t curObject = firstsLocs.size() - 1;
				for (const uint32_t val : queueTriangles)
					linkTriangulationFacesToObjects[val] = curObject;
				currentFirstLocs += locsOfObject.size();
				currentFirstTriangles += trianglesOfObject.size();
				currentFirstOutlines += outlineOfObject.size();
				firstsLocs.push_back(currentFirstLocs);
				firstTriangles.push_back(currentFirstTriangles);
				firstOutlines.push_back(currentFirstOutlines);
				std::copy(locsOfObject.begin(), locsOfObject.end(), std::back_inserter(locsAllObjects));
				std::copy(trianglesOfObject.begin(), trianglesOfObject.end(), std::back_inserter(trianglesAllObjects));
				std::copy(outlineOfObject.begin(), outlineOfObject.end(), std::back_inserter(outlinesAllObjects));
			}
		}
		return locsAllObjects.empty() ? NULL : new ObjectList(xs, ys, _delaunay->getZs() == NULL ? NULL : _delaunay->getZs(), locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, outlinesAllObjects, firstOutlines, linkTriangulationFacesToObjects);
	}*/

	ObjectListInterface* ObjectListFactory::createObjectList2D(DelaunayTriangulationInterface* _delaunay, const std::map <uint32_t, std::vector <uint32_t>>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea)
	{
		const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		std::vector <float> zsTmp;
		if (_delaunay->getZs() == NULL)
			zsTmp = std::vector<float>(_delaunay->nbPoints(), 0.f);
		const float* zs = _delaunay->getZs() == NULL ? zsTmp.data() : _delaunay->getZs();

		ObjectListFactoryInterface::TypeShape type = poca::core::Engine::instance()->getGlobalParameters()["typeObject"].get<ObjectListFactoryInterface::TypeShape>();

		std::vector <uint32_t> locsAllObjects, firstsLocs, firstTriangles, firstOutlines;
		std::vector <poca::core::Vec3mf> trianglesAllObjects, outlinesAllObjects;
		float area = 0.f;

		for (auto it = _selection.begin(); it != _selection.end(); it++) {
			auto indexObj = it->first;
			auto locs = it->second;
			area = 0.f;

			if (_minNbLocs <= locs.size() && locs.size() <= _maxNbLocs && _minArea <= area && area <= _maxArea) {
				firstsLocs.push_back(locsAllObjects.size());
				firstTriangles.push_back(trianglesAllObjects.size());
				firstOutlines.push_back(outlinesAllObjects.size());

				switch (type) {
				case ObjectListFactoryInterface::CONVEX_HULL:
					computeConvexHullObject2D(xs, ys, zs, locs, outlinesAllObjects, trianglesAllObjects, area);
					break;
				default:
					computeConvexHullObject2D(xs, ys, zs, locs, outlinesAllObjects, trianglesAllObjects, area);
					break;
				}

				std::copy(locs.begin(), locs.end(), std::back_inserter(locsAllObjects));
			}
		}
		firstsLocs.push_back(locsAllObjects.size());
		firstTriangles.push_back(trianglesAllObjects.size());
		firstOutlines.push_back(outlinesAllObjects.size());

		return locsAllObjects.empty() ? NULL : new ObjectListDelaunay(xs, ys, _delaunay->getZs() == NULL ? NULL : _delaunay->getZs(), locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, outlinesAllObjects, firstOutlines, std::vector <uint32_t>());

		/*const std::vector<uint32_t>& triangles = _delaunay->getTriangles();
		const poca::core::MyArrayUInt32& neighbors = _delaunay->getNeighbors();
		const std::vector <uint32_t>& tests = neighbors.getFirstElements();
		const std::vector <uint32_t>& testsData = neighbors.getData();

		std::vector <uint32_t> globalSelection(_delaunay->nbFaces(), std::numeric_limits<std::size_t>::max());
		for (auto it = _selection.begin(); it != _selection.end(); it++) {
			auto indexObj = it->first;
			for (auto index : it->second)
				globalSelection[index] = indexObj;
		}

		std::vector <uint32_t> locsAllObjects, firstsLocs, firstTriangles, firstOutlines;
		std::vector <poca::core::Vec3mf> trianglesAllObjects, outlinesAllObjects;
		uint32_t currentFirstLocs = 0, currentFirstTriangles = 0, currentFirstOutlines = 0;
		firstsLocs.push_back(currentFirstLocs);
		firstTriangles.push_back(currentFirstTriangles);
		firstOutlines.push_back(currentFirstOutlines);
		float area = 0.f;

		for (auto it = _selection.begin(); it != _selection.end(); it++) {
			auto indexObj = it->first;
			area = 0.f;
			std::set <uint32_t> locsOfObject, locsOfOutline;
			std::vector <uint32_t> indexTrianglesOfObject;
			std::vector <poca::core::Vec3mf> trianglesOfObject, outlineOfObject;
			for (auto index : it->second) {
				uint32_t i1 = triangles[3 * index], i2 = triangles[3 * index + 1], i3 = triangles[3 * index + 2];
				locsOfObject.insert(i1);
				locsOfObject.insert(i2);
				locsOfObject.insert(i3);
				poca::core::Vec3mf v1(xs[i1], ys[i1], zs[i1]), v2(xs[i2], ys[i2], zs[i2]), v3(xs[i3], ys[i3], zs[i3]);
				indexTrianglesOfObject.push_back(i1);
				indexTrianglesOfObject.push_back(i2);
				indexTrianglesOfObject.push_back(i3);
				float sideA = (v1 - v2).length(), sideB = (v1 - v3).length(), sideC = (v2 - v3).length();
				area += poca::geometry::computeAreaTriangle<float>(sideA, sideB, sideC);

				for (uint32_t i = 0; i < neighbors.nbElementsObject(index); i++) {
					uint32_t indexNeigh = neighbors.elementIObject(index, i);
					if (indexNeigh == std::numeric_limits<std::uint32_t>::max() || globalSelection[indexNeigh] != globalSelection[index]){
						std::array<size_t, 3> edge = _delaunay->getOutline(index, i);
						outlineOfObject.push_back(poca::core::Vec3mf(xs[edge[0]], ys[edge[0]], zs[edge[0]]));
						outlineOfObject.push_back(poca::core::Vec3mf(xs[edge[1]], ys[edge[1]], zs[edge[1]]));
						locsOfOutline.insert(edge[0]);
						locsOfOutline.insert(edge[1]);
					}
				}
			}
			if (_minNbLocs <= locsOfObject.size() && locsOfObject.size() <= _maxNbLocs && _minArea <= area && area <= _maxArea) {
				ObjectListFactoryInterface::TypeShape type = poca::core::Engine::instance()->getGlobalParameters()["typeObject"].get<ObjectListFactoryInterface::TypeShape>();
				switch (type) {
				case ObjectListFactoryInterface::CONVEX_HULL:
					computeConvexHullObject2DFromOutline(xs, ys, zs, locsOfOutline, trianglesOfObject, area);
					break;
				default:
					for (const auto id : indexTrianglesOfObject)
						trianglesOfObject.push_back(poca::core::Vec3mf(xs[id], ys[id], zs[id]));
					break;
				}
				currentFirstLocs += locsOfObject.size();
				currentFirstTriangles += trianglesOfObject.size();
				currentFirstOutlines += outlineOfObject.size();
				firstsLocs.push_back(currentFirstLocs);
				firstTriangles.push_back(currentFirstTriangles);
				firstOutlines.push_back(currentFirstOutlines);
				std::copy(locsOfObject.begin(), locsOfObject.end(), std::back_inserter(locsAllObjects));
				std::copy(trianglesOfObject.begin(), trianglesOfObject.end(), std::back_inserter(trianglesAllObjects));
				std::copy(outlineOfObject.begin(), outlineOfObject.end(), std::back_inserter(outlinesAllObjects));

				for (auto t = 0; t < trianglesOfObject.size(); t += 3)
					std::cout << trianglesOfObject[t] << " - " << trianglesOfObject[t + 1] << " - " << trianglesOfObject[t + 2] << std::endl;
			}
		}
		return locsAllObjects.empty() ? NULL : new ObjectList(xs, ys, _delaunay->getZs() == NULL ? NULL : _delaunay->getZs(), locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, outlinesAllObjects, firstOutlines, globalSelection);*/
	}

	/*ObjectList* ObjectListFactory::createObjectList3D(DelaunayTriangulationInterface* _delaunay, const std::vector <uint32_t>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		const float* zs = _delaunay->getZs();
		const std::vector <float>& volumes = _delaunay->getOriginalHistogram("volume")->getValues();
		const std::vector<uint32_t>& triangles = _delaunay->getTriangles();
		const poca::core::MyArrayUInt32& neighbors = _delaunay->getNeighbors();
		const std::vector <uint32_t> indiceTriangles = neighbors.getFirstElements();

		std::vector <uint32_t> originalSelectionTriangulationFaces(_selection);
		uint32_t debugNbSelect = 0;
		if (!_ROIs.empty()) {
			for (size_t n = 0; n < _delaunay->nbFaces(); n++) {
				if (!originalSelectionTriangulationFaces[n]) continue;
				uint32_t index = indiceTriangles[n];
				uint32_t i1 = triangles[3 * index],
					i2 = triangles[3 * index + 3 * 1],
					i3 = triangles[3 * index + 3 * 2],
					i4 = triangles[3 * index + 3 * 3];
				bool inside = false;
				for (size_t i = 0; i < _ROIs.size() && !inside; i++) {
					bool p1Inside = _ROIs[i]->inside(xs[i1], ys[i1], zs[i1]);
					bool p2Inside = _ROIs[i]->inside(xs[i2], ys[i2], zs[i2]);
					bool p3Inside = _ROIs[i]->inside(xs[i3], ys[i3], zs[i3]);
					bool p4Inside = _ROIs[i]->inside(xs[i4], ys[i4], zs[i4]);
					inside = p1Inside && p2Inside && p3Inside && p4Inside;
				}
				originalSelectionTriangulationFaces[n] = inside;
				if (originalSelectionTriangulationFaces[n]) debugNbSelect++;
			}
		}

		bool applyCutDistance = _dMax != std::numeric_limits < float >::max();
		double dMaxSqr = applyCutDistance ? _dMax * _dMax : _dMax;
		if (applyCutDistance) {
			for (uint32_t n = 0; n < _delaunay->nbFaces(); n++) {
				if (originalSelectionTriangulationFaces[n] == std::numeric_limits<uint32_t>::max() || !applyCutDistance) continue;
				bool kept = true;
				for (uint32_t i = indiceTriangles[n]; i < indiceTriangles[n + 1] && kept; i++) {
					uint32_t i1 = triangles[3 * i], i2 = triangles[3 * i + 1], i3 = triangles[3 * i + 2];
					float d0 = distanceSqr(xs[i1], ys[i1], zs[i1], xs[i2], ys[i2], zs[i2]);
					float d1 = distanceSqr(xs[i2], ys[i2], zs[i2], xs[i3], ys[i3], zs[i3]);
					float d2 = distanceSqr(xs[i3], ys[i3], zs[i3], xs[i1], ys[i1], zs[i1]);
					kept = d0 < dMaxSqr && d1 < dMaxSqr && d2 < dMaxSqr;
				}
				if(!kept)
					originalSelectionTriangulationFaces[n] = std::numeric_limits<uint32_t>::max();
			}
		}
		std::vector <uint32_t> linkTriangulationFacesToObjects(_selection.size(), std::numeric_limits<std::uint32_t>::max());
		std::vector <uint32_t> selectionTriangulationFaces(originalSelectionTriangulationFaces);

		std::vector <uint32_t> locsAllObjects, firstsLocs, firstTriangles, locsAllOutlines, firstOutlineLocs;
		std::vector <poca::core::Vec3mf> trianglesAllObjects;
		uint32_t currentFirstLocs = 0, currentFirstTriangles = 0, currentFirstOutlineLocs = 0;
		firstsLocs.push_back(currentFirstLocs);
		firstTriangles.push_back(currentFirstTriangles);
		firstOutlineLocs.push_back(currentFirstOutlineLocs);
		std::vector <float> volumeObjects;
		float volume = 0.f;
		double volumeD = 0.;
		for (uint32_t n = 0; n < _delaunay->nbFaces(); n++) {
			if (selectionTriangulationFaces[n] == std::numeric_limits<uint32_t>::max()) continue;
			uint32_t indexFoundTriangle = selectionTriangulationFaces[n];
			volume = 0.f;
			volumeD = 0.;
			std::vector <uint32_t> queueTriangles;
			std::set <uint32_t> uniqueTrianglesQueue;
			std::set <uint32_t> locsOfObject, locsOfOutline;
			std::vector <poca::core::Vec3mf> trianglesOfObject;
			queueTriangles.push_back(n);
			uniqueTrianglesQueue.insert(n);
			uint32_t currentTriangle = 0, sizeQueue = queueTriangles.size();
			while (currentTriangle < sizeQueue) {
				uint32_t indexFace = queueTriangles.at(currentTriangle);
				{
					uint32_t index = indiceTriangles[indexFace];
					//Here we have a tetrahedron that is composed of 4 triangles -> 12 vertices
					//If we want to find the 4 vertices, we have to use the first vertex of the four triangles
					//Then we need to determine if a traingle is at the border of the object to add it
					uint32_t is[4] = { triangles[3 * index],
						triangles[3 * index + 3 * 1],
						triangles[3 * index + 3 * 2],
						triangles[3 * index + 3 * 3] };
					for (uint32_t ind : is)
						locsOfObject.insert(ind);

					volume += volumes[indexFace];
					volumeD += volumes[indexFace];

					for (uint32_t i = 0; i < neighbors.nbElementsObject(indexFace); i++) {
						uint32_t indexNeigh = neighbors.elementIObject(indexFace, i);
						if (indexNeigh != std::numeric_limits<std::uint32_t>::max() && selectionTriangulationFaces[indexNeigh] == indexFoundTriangle) {
							//std::cout << "Current " << indexFace << ", adding " << indexNeigh << std::endl;
							if (uniqueTrianglesQueue.find(indexNeigh) == uniqueTrianglesQueue.end()) {
								queueTriangles.push_back(indexNeigh);
								uniqueTrianglesQueue.insert(indexNeigh);
								selectionTriangulationFaces[indexNeigh] = std::numeric_limits<std::uint32_t>::max();
							}
						}
						if (indexNeigh != std::numeric_limits<std::uint32_t>::max() && originalSelectionTriangulationFaces[indexNeigh] != indexFoundTriangle) {
							uint32_t indexN = indiceTriangles[indexNeigh];
							uint32_t isN[4] = { triangles[3 * indexN],
								triangles[3 * indexN + 3 * 1],
								triangles[3 * indexN + 3 * 2],
								triangles[3 * indexN + 3 * 3] };
							std::vector <uint32_t> indexCurAndNeighLocs = { is[0], is[1], is[2], is[3], isN[0], isN[1], isN[2], isN[3] };
							std::map <uint32_t, int> duplicates;
							poca::core::findDuplicates(indexCurAndNeighLocs, duplicates);
							if (duplicates.size() != 3)
								std::cout << "Seems to have a problem" << std::endl;
							else {
								//try to keep the same orientation (cw or ccw) for all triangles
								//To achieve that, we compute the normal of the triangle
								//and compare to the vector coming from one point of the triangle to the centroid of the tetrahedron
								//if their dot product is positive, they are having the same orientation: we change the order of the triangle vertices
								//if it's negative, no change is needed
								poca::core::Vec3mf centroidN;
								for (uint32_t idTmp : is)
									centroidN += (poca::core::Vec3mf(xs[idTmp], ys[idTmp], zs[idTmp]) / 4.f);
								poca::core::Vec3mf vs[3];
								size_t cptt = 0;
								for (std::map <uint32_t, int>::const_iterator it = duplicates.begin(); it != duplicates.end(); it++, cptt++) {
									vs[cptt].set(xs[it->first], ys[it->first], zs[it->first]);
									locsOfOutline.insert(it->first);
								}
								poca::core::Vec3mf e1 = vs[1] - vs[0], e2 = vs[2] - vs[0], e3 = centroidN - vs[0], normal = e1.cross(e2);
								normal.normalize();
								e3.normalize();
								if (normal.dot(e3) < 0.f)
									std::reverse(std::begin(vs), std::end(vs));
								for (const poca::core::Vec3mf& v : vs)
									trianglesOfObject.push_back(v);
							}
						}
						if (indexNeigh == std::numeric_limits<std::uint32_t>::max()) {
							poca::core::Vec3mf centroidN;
							for (uint32_t idTmp : is)
								centroidN += (poca::core::Vec3mf(xs[idTmp], ys[idTmp], zs[idTmp]) / 4.f);
							for (uint32_t k = 0; k < 4; k++) {
								uint32_t cur = is[k], next = is[(k + 1) % 4], nnext = is[(k + 2) % 4];
								poca::core::Vec3mf vs[3] = { poca::core::Vec3mf(xs[cur], ys[cur], zs[cur]), poca::core::Vec3mf(xs[next], ys[next], zs[next]), poca::core::Vec3mf(xs[nnext], ys[nnext], zs[nnext]) };
								poca::core::Vec3mf e1 = vs[1] - vs[0], e2 = vs[2] - vs[0], e3 = centroidN - vs[0], normal = e1.cross(e2);
								normal.normalize();
								e3.normalize();
								if (normal.dot(e3) < 0.f)
									std::reverse(std::begin(vs), std::end(vs));
								for (const poca::core::Vec3mf& v : vs)
									trianglesOfObject.push_back(v);
								locsOfOutline.insert(cur); locsOfOutline.insert(next); locsOfOutline.insert(nnext);
							}
						}
					}
					sizeQueue = queueTriangles.size();
				}
				currentTriangle++;
			}
			if (_minNbLocs <= locsOfObject.size() && locsOfObject.size() <= _maxNbLocs && _minArea <= volume && volume <= _maxArea) {
				size_t curObject = firstsLocs.size() - 1;
				for (const uint32_t val : queueTriangles)
					linkTriangulationFacesToObjects[val] = curObject;
				currentFirstLocs += locsOfObject.size();
				currentFirstTriangles += trianglesOfObject.size();
				firstsLocs.push_back(currentFirstLocs);
				firstTriangles.push_back(currentFirstTriangles);
				std::copy(locsOfObject.begin(), locsOfObject.end(), std::back_inserter(locsAllObjects));
				std::copy(trianglesOfObject.begin(), trianglesOfObject.end(), std::back_inserter(trianglesAllObjects));
				volumeObjects.push_back(volume);
				currentFirstOutlineLocs += locsOfOutline.size();
				firstOutlineLocs.push_back(currentFirstOutlineLocs);
				std::copy(locsOfOutline.begin(), locsOfOutline.end(), std::back_inserter(locsAllOutlines));
			}
		}
		return locsAllObjects.empty() ? NULL : new ObjectList(xs, ys, zs, locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, volumeObjects, linkTriangulationFacesToObjects, locsAllOutlines, firstOutlineLocs);
	}*/

	ObjectListInterface* ObjectListFactory::createObjectList3D(DelaunayTriangulationInterface* _delaunay, const std::map <uint32_t, std::vector <uint32_t>>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea)
	{
		const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		const float* zs = _delaunay->getZs();

		ObjectListFactoryInterface::TypeShape type = poca::core::Engine::instance()->getGlobalParameters()["typeObject"].get<ObjectListFactoryInterface::TypeShape>();

		std::vector <uint32_t> locsAllObjects, firstsLocs, firstTriangles, firstOutlines;
		std::vector <poca::core::Vec3mf> trianglesAllObjects, outlinesAllObjects, triCHull;
		float area = 0.f;
		std::vector <float> volumeObjects;
		std::vector < Surface_mesh_3_double> meshes;

		for (auto it = _selection.begin(); it != _selection.end(); it++) {
			auto indexObj = it->first;
			auto locs = it->second;
			area = 0.f;

			if (_minNbLocs <= locs.size() && locs.size() <= _maxNbLocs && _minArea <= area && area <= _maxArea) {
				firstsLocs.push_back(locsAllObjects.size());
				firstTriangles.push_back(trianglesAllObjects.size());
				firstOutlines.push_back(outlinesAllObjects.size());

				meshes.push_back(Surface_mesh_3_double());

				switch (type) {
				case ObjectListFactoryInterface::MESH:
					computeConvexHullObject3DMesh(xs, ys, zs, locs, meshes.back(), area);
					break;
				default:
					computeConvexHullObject3D(xs, ys, zs, locs, outlinesAllObjects, triCHull, area);
					break;
				}

				std::copy(locs.begin(), locs.end(), std::back_inserter(locsAllObjects));
				std::copy(triCHull.begin(), triCHull.end(), std::back_inserter(trianglesAllObjects));
				volumeObjects.push_back(area);
			}
		}
		//return new poca::geometry::ObjectListMesh(meshes);
		firstsLocs.push_back(locsAllObjects.size());
		firstTriangles.push_back(trianglesAllObjects.size());
		firstOutlines.push_back(outlinesAllObjects.size());

		switch (type) {
		case ObjectListFactoryInterface::MESH:
			return locsAllObjects.empty() ? NULL : new poca::geometry::ObjectListMesh(meshes);
			break;
		default:
			return locsAllObjects.empty() ? NULL : new ObjectListDelaunay(xs, ys, zs, locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, volumeObjects, std::vector<uint32_t>(), locsAllObjects, firstsLocs, std::vector <poca::core::Vec3mf>());
			break;
		}
		return NULL;
		//return locsAllObjects.empty() ? NULL : new ObjectListDelaunay(xs, ys, zs, locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, volumeObjects, std::vector<uint32_t>(), locsAllObjects, firstsLocs, std::vector <poca::core::Vec3mf>());
		
		/*const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		const float* zs = _delaunay->getZs();
		const std::vector <float>& volumes = static_cast<poca::core::Histogram<float>*>(_delaunay->getOriginalHistogram("volume"))->getValues();
		const std::vector<uint32_t>& triangles = _delaunay->getTriangles();
		const poca::core::MyArrayUInt32& neighbors = _delaunay->getNeighbors();
		const std::vector <uint32_t> indiceTriangles = neighbors.getFirstElements();

		std::vector <uint32_t> originalSelectionTriangulationFaces(_delaunay->nbFaces(), std::numeric_limits<uint32_t>::max());
		for (auto it = _selection.begin(); it != _selection.end(); it++)
			for (auto indexFace : it->second)
				originalSelectionTriangulationFaces[indexFace] = it->first;

		bool applyCutDistance = _dMax != std::numeric_limits < float >::max();
		double dMaxSqr = applyCutDistance ? _dMax * _dMax : _dMax;
		if (applyCutDistance) {
			for (uint32_t n = 0; n < _delaunay->nbFaces(); n++) {
				if (originalSelectionTriangulationFaces[n] == std::numeric_limits<uint32_t>::max() || !applyCutDistance) continue;
				bool kept = true;
				for (uint32_t i = indiceTriangles[n]; i < indiceTriangles[n + 1] && kept; i++) {
					uint32_t i1 = triangles[3 * i], i2 = triangles[3 * i + 1], i3 = triangles[3 * i + 2];
					float d0 = distanceSqr(xs[i1], ys[i1], zs[i1], xs[i2], ys[i2], zs[i2]);
					float d1 = distanceSqr(xs[i2], ys[i2], zs[i2], xs[i3], ys[i3], zs[i3]);
					float d2 = distanceSqr(xs[i3], ys[i3], zs[i3], xs[i1], ys[i1], zs[i1]);
					kept = d0 < dMaxSqr&& d1 < dMaxSqr&& d2 < dMaxSqr;
				}
				if (!kept)
					originalSelectionTriangulationFaces[n] = std::numeric_limits<uint32_t>::max();
			}
		}

		std::vector <bool> selecTmp(originalSelectionTriangulationFaces.size());
		for (auto n = 0; n < originalSelectionTriangulationFaces.size(); n++)
			selecTmp[n] = originalSelectionTriangulationFaces[n] != std::numeric_limits<uint32_t>::max();
		_delaunay->setSelection(selecTmp);
		//_delaunay->executeCommand(false, "updateFeature");

		std::vector <uint32_t> linkTriangulationFacesToObjects(originalSelectionTriangulationFaces.size(), std::numeric_limits<std::uint32_t>::max());
		std::vector <uint32_t> selectionTriangulationFaces(originalSelectionTriangulationFaces);

		std::vector <uint32_t> locsAllObjects, firstsLocs, firstTriangles, locsAllOutlines, firstOutlineLocs;
		std::vector <poca::core::Vec3mf> trianglesAllObjects, normalsAllOutlineLocs;
		uint32_t currentFirstLocs = 0, currentFirstTriangles = 0, currentFirstOutlineLocs = 0;
		firstsLocs.push_back(currentFirstLocs);
		firstTriangles.push_back(currentFirstTriangles);
		firstOutlineLocs.push_back(currentFirstOutlineLocs);
		std::vector <float> volumeObjects;
		float volume = 0.f;
		double volumeD = 0.;

		std::vector <uint32_t> allIndexesTriangles;

		//For ObjectListMesh
		std::vector <std::vector <poca::core::Vec3mf>> meshPoints;
		std::vector <std::vector <std::vector <std::size_t>>> meshTris;
		std::vector < Surface_mesh_3_double> meshes;

		for (uint32_t n = 0; n < _delaunay->nbFaces(); n++) {
			if (selectionTriangulationFaces[n] == std::numeric_limits<uint32_t>::max()) continue;
			uint32_t indexFoundTriangle = selectionTriangulationFaces[n];
			volume = 0.f;
			volumeD = 0.;
			std::vector <uint32_t> queueTriangles, indexTrianglesOfObject;
			std::set <uint32_t> locsOfObject, locsOfOutline;
			std::vector <poca::core::Vec3mf> trianglesOfObject, normalsTrianglesOfObject, normalOutlineLocObject;
			queueTriangles.push_back(n);
			selectionTriangulationFaces[n] = std::numeric_limits<uint32_t>::max();
			uint32_t currentTriangle = 0, sizeQueue = queueTriangles.size();
			while (currentTriangle < sizeQueue) {
				uint32_t indexFace = queueTriangles.at(currentTriangle);
				{

					uint32_t index = indiceTriangles[indexFace];
					//Here we have a tetrahedron that is composed of 4 triangles -> 12 vertices
					//If we want to find the 4 vertices, we have to use the first vertex of the four triangles
					//Then we need to determine if a traingle is at the border of the object to add it
					uint32_t is[4] = { triangles[3 * index],
						triangles[3 * index + 3 * 1],
						triangles[3 * index + 3 * 2],
						triangles[3 * index + 3 * 3] };
					for (uint32_t ind : is)
						locsOfObject.insert(ind);

					volume += volumes[indexFace];
					volumeD += volumes[indexFace];

					for (uint32_t i = 0; i < neighbors.nbElementsObject(indexFace); i++) {
						uint32_t indexNeigh = neighbors.elementIObject(indexFace, i);
						if (indexNeigh == std::numeric_limits<std::uint32_t>::max()) {
							poca::core::Vec3mf centroidN;
							for (uint32_t idTmp : is)
								centroidN += (poca::core::Vec3mf(xs[idTmp], ys[idTmp], zs[idTmp]) / 4.f);
							uint32_t ids[3] = { is[(i + 1) % 4] , is[(i + 2) % 4], is[(i + 3) % 4] };
							poca::core::Vec3mf vs[3] = { poca::core::Vec3mf(xs[ids[0]], ys[ids[0]], zs[ids[0]]), poca::core::Vec3mf(xs[ids[1]], ys[ids[1]], zs[ids[1]]), poca::core::Vec3mf(xs[ids[2]], ys[ids[2]], zs[ids[2]]) }, centroidF(0, 0, 0);
							poca::core::Vec3mf e1 = vs[1] - vs[0], e2 = vs[2] - vs[0], e3 = centroidN - vs[0], normal = e1.cross(e2);
							normal.normalize();
							e3.normalize();
							if (normal.dot(e3) < 0.f)
								std::reverse(std::begin(ids), std::end(ids));
							else
								normal = -normal;
							for (const uint32_t id : ids) {
								indexTrianglesOfObject.push_back(id);
								locsOfOutline.insert(id);
								centroidF += poca::core::Vec3mf(xs[id], ys[id], zs[id]) / 3.f;
							}
							normalsTrianglesOfObject.push_back(normal);
						}
						else if (originalSelectionTriangulationFaces[indexNeigh] != indexFoundTriangle) {
							uint32_t indexN = indiceTriangles[indexNeigh];
							uint32_t isN[4] = { triangles[3 * indexN],
								triangles[3 * indexN + 3 * 1],
								triangles[3 * indexN + 3 * 2],
								triangles[3 * indexN + 3 * 3] };
							std::vector <uint32_t> indexCurAndNeighLocs = { is[0], is[1], is[2], is[3], isN[0], isN[1], isN[2], isN[3] };
							std::map <uint32_t, int> duplicates;
							poca::core::findDuplicates(indexCurAndNeighLocs, duplicates);
							if (duplicates.size() != 3)
								std::cout << "Seems to have a problem" << std::endl;
							else {
								//try to keep the same orientation (cw or ccw) for all triangles
								//To achieve that, we compute the normal of the triangle
								//and compare to the vector coming from one point of the triangle to the centroid of the tetrahedron
								//if their dot product is positive, they are having the same orientation: we change the order of the triangle vertices
								//if it's negative, no change is needed
								poca::core::Vec3mf centroidN;
								for (uint32_t idTmp : is)
									centroidN += (poca::core::Vec3mf(xs[idTmp], ys[idTmp], zs[idTmp]) / 4.f);
								uint32_t ids[3];
								poca::core::Vec3mf vs[3], centroidF(0, 0, 0);
								size_t cptt = 0;
								for (std::map <uint32_t, int>::const_iterator it = duplicates.begin(); it != duplicates.end(); it++, cptt++) {
									vs[cptt].set(xs[it->first], ys[it->first], zs[it->first]);
									ids[cptt] = it->first;
								}
								poca::core::Vec3mf e1 = vs[1] - vs[0], e2 = vs[2] - vs[0], e3 = centroidN - vs[0], normal = e1.cross(e2);
								normal.normalize();
								e3.normalize();
								if (normal.dot(e3) < 0.f)
									std::reverse(std::begin(ids), std::end(ids));
								else
									normal = -normal;
								for (const uint32_t id : ids) {
									indexTrianglesOfObject.push_back(id);
									locsOfOutline.insert(id);
									centroidF += poca::core::Vec3mf(xs[id], ys[id], zs[id]) / 3.f;
								}
								normalsTrianglesOfObject.push_back(normal);
							}
						}
						else if (selectionTriangulationFaces[indexNeigh] == indexFoundTriangle) {
							queueTriangles.push_back(indexNeigh);
							selectionTriangulationFaces[indexNeigh] = std::numeric_limits<uint32_t>::max();
						}
					}
					sizeQueue = queueTriangles.size();
				}
				currentTriangle++;
			}

			if (_minNbLocs <= locsOfObject.size() && locsOfObject.size() <= _maxNbLocs && _minArea <= volume && volume <= _maxArea) {
				ObjectListFactoryInterface::TypeShape type = poca::core::Engine::instance()->getGlobalParameters()["typeObject"].get<ObjectListFactoryInterface::TypeShape>();
				switch (type) {
				case ObjectListFactoryInterface::TRIANGULATION:
					for (const auto id : indexTrianglesOfObject)
						trianglesOfObject.push_back(poca::core::Vec3mf(xs[id], ys[id], zs[id]));
					break;
				case ObjectListFactoryInterface::CONVEX_HULL:
					computeConvexHullObject3DFromOutline(xs, ys, zs, locsOfOutline, trianglesOfObject, volume);
					break;
				case ObjectListFactoryInterface::POISSON_SURFACE:
					computePoissonSurfaceObject(xs, ys, zs, locsOfOutline, indexTrianglesOfObject, normalsTrianglesOfObject, trianglesOfObject, volume);
					break;
				case ObjectListFactoryInterface::ALPHA_SHAPE:
					computeAlphaShape(xs, ys, zs, locsOfOutline, trianglesOfObject, volume);
					break;
				default:
					break;
				}

				size_t curObject = firstsLocs.size() - 1;
				for (const uint32_t val : queueTriangles) {
					linkTriangulationFacesToObjects[val] = curObject;
				}
				currentFirstLocs += locsOfObject.size();
				firstsLocs.push_back(currentFirstLocs);
				std::copy(locsOfObject.begin(), locsOfObject.end(), std::back_inserter(locsAllObjects));

				currentFirstOutlineLocs += locsOfOutline.size();
				firstOutlineLocs.push_back(currentFirstOutlineLocs);
				std::copy(locsOfOutline.begin(), locsOfOutline.end(), std::back_inserter(locsAllOutlines));

				currentFirstTriangles += trianglesOfObject.size();
				firstTriangles.push_back(currentFirstTriangles);
				std::copy(trianglesOfObject.begin(), trianglesOfObject.end(), std::back_inserter(trianglesAllObjects));

				computeNormalOfLocsObject(locsOfOutline, indexTrianglesOfObject, normalsTrianglesOfObject, normalOutlineLocObject);
				std::copy(normalOutlineLocObject.begin(), normalOutlineLocObject.end(), std::back_inserter(normalsAllOutlineLocs));

				volumeObjects.push_back(volume);

				std::copy(indexTrianglesOfObject.begin(), indexTrianglesOfObject.end(), std::back_inserter(allIndexesTriangles));
				
				//For ObjectListMesh
				thrust::host_vector <uint32_t> h_triangles(indexTrianglesOfObject), h_data(indexTrianglesOfObject);
				//std::copy(h_data.begin(), h_data.end(), std::ostream_iterator<uint32_t>(std::cout, " "));
				//std::cout << std::endl;
				thrust::sort(thrust::host, h_triangles.begin(), h_triangles.end());
				thrust::host_vector<uint32_t> h_unique = h_triangles;
				const auto end = thrust::unique(h_unique.begin(), h_unique.end());
				auto nbVertices = thrust::distance(h_unique.begin(), end);

				meshPoints.push_back(std::vector <poca::core::Vec3mf>());
				meshPoints.back().resize(nbVertices);
				for (auto i = 0; i < nbVertices; i++) {
					meshPoints.back()[i].set(xs[h_unique[i]], ys[h_unique[i]], zs[h_unique[i]]);
				}

				//relabel the faces
				thrust::lower_bound(h_unique.begin(), end, h_data.begin(), h_data.end(), h_triangles.begin());
				meshTris.push_back(std::vector <std::vector <std::size_t>>());
				for (auto i = 0; i < h_triangles.size(); i += 3) {
					meshTris.back().push_back(std::vector<std::size_t>{h_triangles[i + 2], h_triangles[i + 1], h_triangles[i]});
				}
			}
		}

		ObjectListInterface* objs;
		ObjectListFactoryInterface::TypeShape type = poca::core::Engine::instance()->getGlobalParameters()["typeObject"].get<ObjectListFactoryInterface::TypeShape>();
		if (type == ObjectListFactoryInterface::POISSON_SURFACE) {
			objs = locsAllObjects.empty() ? NULL : new poca::geometry::ObjectListMesh(meshes);
		}
		else if (type == ObjectListFactoryInterface::TRIANGULATION) {
			objs = locsAllObjects.empty() ? NULL : new ObjectListDelaunay(xs, ys, zs, locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, volumeObjects, linkTriangulationFacesToObjects, locsAllOutlines, firstOutlineLocs, normalsAllOutlineLocs);
		}
		else {
			double targetLength = poca::core::Engine::instance()->getGlobalParameters()["meshTargetLength"].get<double>();
			int iterations = poca::core::Engine::instance()->getGlobalParameters()["meshIterations"].get<int>();

			std::vector <poca::core::ROIInterface*> ROIs;
			objs = locsAllObjects.empty() ? NULL : new poca::geometry::ObjectListMesh(meshPoints, meshTris, ROIs, true, true, targetLength, iterations);
		}

		return objs;*/
	}

	void ObjectListFactory::computeConvexHullObject2D(const float* _xs, const float* _ys, const float* _zs, const std::vector <uint32_t>& _locs, std::vector <poca::core::Vec3mf>& _outlineLocs, std::vector <poca::core::Vec3mf>& _triangles, float& _feature)
	{
		_feature = 0.f;
		std::vector <double>* coords = new std::vector <double>();
		coords->resize(_locs.size() * 2);
		uint32_t cpt = 0;
		for (auto index : _locs) {
			(*coords)[2 * cpt] = _xs[index];
			(*coords)[2 * cpt + 1] = _ys[index];
			cpt++;
		}
		delaunator::Delaunator* d = new delaunator::Delaunator(*coords);
		for (const auto& index : d->triangles) {
			_triangles.push_back(poca::core::Vec3mf((float)(*coords)[2 * index], (float)(*coords)[2 * index + 1], 0));
		}
		_feature = d->get_hull_area();
		std::vector <std::uint32_t> outline;
		d->get_outline_edges(outline);
		for (auto n = 0; n < outline.size(); n++) {
			auto index = _locs[outline[n]];
			_outlineLocs.push_back(poca::core::Vec3mf(_xs[index], _ys[index], _zs[index]));
		}
		delete d;
		delete coords;
	}

	void ObjectListFactory::computeConvexHullObject3D(const float* _xs, const float* _ys, const float* _zs, const std::vector <uint32_t>& _locs, std::vector <poca::core::Vec3mf>& _outlineLocs, std::vector <poca::core::Vec3mf>& _triangles, float& _feature)
	{
		std::vector <Point_3_inexact> points;
		for (const auto id : _locs)
			points.push_back(Point_3_inexact(_xs[id], _ys[id], _zs[id]));

		Polyhedron_3_inexact poly;
		CGAL::convex_hull_3(points.begin(), points.end(), poly);

		_triangles.clear();
		for (Polyhedron_3_inexact::Facet_const_iterator fi = poly.facets_begin(); fi != poly.facets_end(); fi++) {
			Polyhedron_3_inexact::Halfedge_around_facet_const_circulator hfc = fi->facet_begin();
			poca::core::Vec3mf prec;
			bool firstDone = false;
			do {
				Polyhedron_3_inexact::Halfedge_const_handle hh = hfc;
				Polyhedron_3_inexact::Vertex_const_handle v = hh->vertex();
				_triangles.insert(_triangles.begin(), poca::core::Vec3mf(CGAL::to_double(v->point().x()), CGAL::to_double(v->point().y()), CGAL::to_double(v->point().z())));
			} while (++hfc != fi->facet_begin());
		}
		_feature = CGAL::Polygon_mesh_processing::volume(poly);
	}

	void ObjectListFactory::computeConvexHullObject3DMesh(const float* _xs, const float* _ys, const float* _zs, const std::vector <uint32_t>& _locs, Surface_mesh_3_double& _mesh, float& _feature)
	{
		std::vector <Point_3_inexact> points;
		for (const auto id : _locs)
			points.push_back(Point_3_inexact(_xs[id], _ys[id], _zs[id]));

		Polyhedron_3_inexact poly;
		CGAL::convex_hull_3(points.begin(), points.end(), poly);

		CGAL::copy_face_graph(poly, _mesh);
		assert(CGAL::is_valid_polygon_mesh(_mesh));
	
		_feature = CGAL::Polygon_mesh_processing::volume(poly);
	}

	void ObjectListFactory::computeConvexHullObject2DFromOutline(const float* _xs, const float* _ys, const float* _zs, const std::set <uint32_t>& _locs, std::vector <poca::core::Vec3mf>& _triangles, float& _feature)
	{
		_feature = 0.f;
		std::vector <double>* coords = new std::vector <double>();
		coords->resize(_locs.size() * 2);
		uint32_t cpt = 0;
		for(auto index : _locs) {
			(*coords)[2 * cpt] = _xs[index];
			(*coords)[2 * cpt + 1] = _ys[index];
			cpt++;
		}
		delaunator::Delaunator* d = new delaunator::Delaunator(*coords);
		for (const auto& index : d->triangles) {
			_triangles.push_back(poca::core::Vec3mf((float)(*coords)[2 * index], (float)(*coords)[2 * index + 1], 0));
		}
		_feature = d->get_hull_area();
		delete d;
		delete coords;
	}

	void ObjectListFactory::computeConvexHullObject3DFromOutline(const float* _xs, const float* _ys, const float* _zs, const std::set <uint32_t>& _locs, std::vector <poca::core::Vec3mf>& _triangles, float& _feature)
	{
		std::vector <Point_3_inexact> points;
		for (const auto id : _locs)
			points.push_back(Point_3_inexact(_xs[id], _ys[id], _zs[id]));

		Polyhedron_3_inexact poly;
		CGAL::convex_hull_3(points.begin(), points.end(), poly);

		_triangles.clear();
		for (Polyhedron_3_inexact::Facet_const_iterator fi = poly.facets_begin(); fi != poly.facets_end(); fi++) {
			Polyhedron_3_inexact::Halfedge_around_facet_const_circulator hfc = fi->facet_begin();
			poca::core::Vec3mf prec;
			bool firstDone = false;
			do {
				Polyhedron_3_inexact::Halfedge_const_handle hh = hfc;
				Polyhedron_3_inexact::Vertex_const_handle v = hh->vertex();
				_triangles.insert(_triangles.begin(), poca::core::Vec3mf(CGAL::to_double(v->point().x()), CGAL::to_double(v->point().y()), CGAL::to_double(v->point().z())));
			} while (++hfc != fi->facet_begin());
		}
		_feature = CGAL::Polygon_mesh_processing::volume(poly);
	}

#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
	typedef K_inexact::FT FT;
	typedef CGAL::First_of_pair_property_map<pointWnormal_3_inexact> Point_map;
	typedef CGAL::Second_of_pair_property_map<pointWnormal_3_inexact> Normal_map;
	typedef K_inexact::Sphere_3 Sphere;
	typedef std::vector<pointWnormal_3_inexact> PointList;
	typedef CGAL::Polyhedron_3<K_inexact> Polyhedron;
	typedef CGAL::Poisson_reconstruction_function<K_inexact> Poisson_reconstruction_function;
	typedef CGAL::Implicit_surface_3<K_inexact, Poisson_reconstruction_function> Surface_3;
	namespace params = CGAL::parameters;

	template<typename Concurrency_tag, typename PointSet>
	void poisson_reconstruction(const PointSet& points, Polyhedron& output_mesh)
	{
		typedef CGAL::Labeled_mesh_domain_3<K_inexact> Mesh_domain;
		typedef CGAL::Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
		typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
		typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

		// Poisson options
		//FT sm_angle = 20.0; // Min triangle angle in degrees.
		//FT sm_radius = 100; // Max triangle size w.r.t. point set average spacing.
		//FT sm_distance = 0.25; // Surface Approximation error w.r.t. point set average spacing.

		FT sm_angle = poca::core::Engine::instance()->getGlobalParameters()["angle"].get<double>();
		FT sm_radius = poca::core::Engine::instance()->getGlobalParameters()["radius"].get<double>();
		FT sm_distance = poca::core::Engine::instance()->getGlobalParameters()["distance"].get<double>();
		FT sm_factorAverageSpacing = poca::core::Engine::instance()->getGlobalParameters()["factorAverageSpacing"].get<double>();

		CGAL::Timer time;
		time.start();

		CGAL::Timer total_time;
		total_time.start();

		// Creates implicit function from the read points using the default solver.

		// Note: this method requires an iterator over points
		// + property maps to access each point's position and normal.
		Poisson_reconstruction_function function(points.begin(), points.end(), Point_map(), Normal_map());

		// Computes the Poisson indicator function f()
		// at each vertex of the triangulation.
		if (!function.compute_implicit_function())
		{
			std::cerr << "compute_implicit_function() failed." << std::endl;
			return;
		}

		time.stop();
		std::cout << "Compute_implicit_function : " << time.time() << " seconds." << std::endl;
		time.reset();
		time.start();

		// Computes average spacing
		FT average_spacing = CGAL::compute_average_spacing<Concurrency_tag>(points, 6 /* knn = 1 ring */, CGAL::parameters::point_map(Point_map()));
		average_spacing /= sm_factorAverageSpacing;

		time.stop();
		std::cout << "Average spacing : " << time.time() << " seconds." << std::endl;
		time.reset();
		time.start();

		// Gets one point inside the implicit surface
		// and computes implicit function bounding sphere radius.
		Point_3_inexact inner_point = function.get_inner_point();
		Sphere bsphere = function.bounding_sphere();
		FT radius = std::sqrt(bsphere.squared_radius());

		// Defines the implicit surface: requires defining a
		// conservative bounding sphere centered at inner point.
		FT sm_sphere_radius = 5.0 * radius;
		FT sm_dichotomy_error = sm_distance * average_spacing / 1000.0; // Dichotomy error must be << sm_distance
		std::cout << "dichotomy error = " << sm_dichotomy_error << std::endl;
		std::cout << "sm_dichotomy_error / sm_sphere_radius = " << sm_dichotomy_error / sm_sphere_radius << std::endl;

		Sphere sm_sphere(inner_point, sm_sphere_radius * sm_sphere_radius);

		Surface_3 surface(function,
			sm_sphere,
			sm_dichotomy_error / sm_sphere_radius);

		time.stop();
		std::cout << "Surface created in " << time.time() << " seconds." << std::endl;
		time.reset();
		time.start();

		// Defines surface mesh generation criteria
		CGAL::Mesh_criteria_3<Tr> criteria(params::facet_angle = sm_angle,
			params::facet_size = sm_radius * average_spacing,
			params::facet_distance = sm_distance * average_spacing);

		Mesh_domain domain = Mesh_domain::create_implicit_mesh_domain(surface, sm_sphere,
			params::relative_error_bound(sm_dichotomy_error / sm_sphere_radius));

		// Generates surface mesh with manifold option
		std::cout << "Start meshing...";
		std::cout.flush();
		C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, params::no_exude().no_perturb().manifold_with_boundary());
		const auto& tr = c3t3.triangulation();

		time.stop();
		std::cout << "\nTet mesh created in " << time.time() << " seconds." << std::endl;
		time.reset();
		time.start();

		if (tr.number_of_vertices() == 0)
		{
			std::cerr << "Triangulation empty!" << std::endl;
			return;
		}

		// saves reconstructed surface mesh
		CGAL::facets_in_complex_3_to_triangle_mesh(c3t3, output_mesh);

		time.stop();
		std::cout << "Surface extracted in " << time.time() << " seconds." << std::endl;
		time.reset();
		time.start();

		total_time.stop();
		std::cout << "Total time : " << total_time.time() << " seconds." << std::endl;
	}
#endif

	void ObjectListFactory::computePoissonSurfaceObject(const float* _xs, const float* _ys, const float* _zs, const std::set <uint32_t>& _locs, const std::vector <uint32_t>& _trianglesIndexes, const std::vector <poca::core::Vec3mf>& _normals, std::vector <poca::core::Vec3mf>& _triangles, float& _volume)
	{
		auto maxIndex = *std::max_element(_locs.begin(), _locs.end());
		std::vector <poca::core::Vec3mf> normalPerLoc;

		std::cout << __LINE__ << " - " << _locs.size() << std::endl;
		
		computeNormalOfLocsObject(_locs, _trianglesIndexes, _normals, normalPerLoc);
		std::cout << __LINE__ << std::endl;
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
		PointList points;
		size_t cpt = 0;
		for (auto it = _locs.begin(); it != _locs.end(); it++, cpt++) {
			auto id = *it;
			points.push_back(std::make_pair(Point_3_inexact(_xs[id], _ys[id], _zs[id]), Vector_3_inexact((double)normalPerLoc[cpt].x(), (double)normalPerLoc[cpt].y(), (double)normalPerLoc[cpt].z())));
		}
		std::cout << "\n\n### Parallel mode ###" << std::endl;
		Polyhedron poly;
		poisson_reconstruction<CGAL::Parallel_tag>(points, poly);
		_triangles.clear();
		for (Polyhedron::Facet_const_iterator fi = poly.facets_begin(); fi != poly.facets_end(); fi++) {
			Polyhedron::Halfedge_around_facet_const_circulator hfc = fi->facet_begin();
			poca::core::Vec3mf prec;
			bool firstDone = false;
			do {
				Polyhedron::Halfedge_const_handle hh = hfc;
				Polyhedron::Vertex_const_handle v = hh->vertex();
				_triangles.push_back(poca::core::Vec3mf(CGAL::to_double(v->point().x()), CGAL::to_double(v->point().y()), CGAL::to_double(v->point().z())));
			} while (++hfc != fi->facet_begin());
		}
		/*Point_set point_set(true);
		size_t cpt = 0;
		for (auto it = _locs.begin(); it != _locs.end(); it++, cpt++) {
			auto id = *it;
			point_set.insert(Point_3_inexact(_xs[id], _ys[id], _zs[id]), Vector_3_inexact((double)-normalPerLoc[cpt].x(), (double)-normalPerLoc[cpt].y(), (double)-normalPerLoc[cpt].z()));
		}
		std::cout << __LINE__ << std::endl;
		CGAL::Surface_mesh<Point_3_inexact> output_mesh;
		double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(point_set, 6);
		average_spacing *= 3;

		std::cout << __LINE__ << " - " << average_spacing << std::endl;
		if (CGAL::poisson_surface_reconstruction_delaunay(point_set.begin(), point_set.end(), point_set.point_map(), point_set.normal_map(), output_mesh, average_spacing))
		{
			std::cout << __LINE__ << std::endl;
			_triangles.clear();
			for (CGAL::Surface_mesh<Point_3_inexact>::Face_index fd : output_mesh.faces()) {
				int j = 0; 
				CGAL::Vertex_around_face_iterator<CGAL::Surface_mesh<Point_3_inexact>> vbegin, vend;
				for (boost::tie(vbegin, vend) = vertices_around_face(output_mesh.halfedge(fd), output_mesh); vbegin != vend; vbegin++) {
					j++;
					auto p = output_mesh.point(*vbegin);
					_triangles.push_back(poca::core::Vec3mf(p.x(), p.y(), p.z()));
				}
			}
			std::cout << __LINE__ << std::endl;
		}
		else
			std::cout << "ERROR !!!!!!!!!!!!!!!!" << std::endl;*/
#else
		std::vector<pointWnormal_3_inexact> points;
		size_t cpt = 0;
		for (auto it = _locs.begin(); it != _locs.end(); it++, cpt++) {
			auto id = *it;
			points.push_back(std::make_pair(Point_3_inexact(_xs[id], _ys[id], _zs[id]), Vector_3_inexact((double)normalPerLoc[cpt].x(), (double)normalPerLoc[cpt].y(), (double)normalPerLoc[cpt].z())));
		}

		Polyhedron_3_inexact poly;
		double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(points, 6, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<pointWnormal_3_inexact>()));
		//average_spacing /= 3;

		if (CGAL::poisson_surface_reconstruction_delaunay(points.begin(), points.end(),	CGAL::First_of_pair_property_map<pointWnormal_3_inexact>(), CGAL::Second_of_pair_property_map<pointWnormal_3_inexact>(), poly, average_spacing))
		{
			_triangles.clear();
			for (Polyhedron_3_inexact::Facet_const_iterator fi = poly.facets_begin(); fi != poly.facets_end(); fi++) {
				Polyhedron_3_inexact::Halfedge_around_facet_const_circulator hfc = fi->facet_begin();
				poca::core::Vec3mf prec;
				bool firstDone = false;
				do {
					Polyhedron_3_inexact::Halfedge_const_handle hh = hfc;
					Polyhedron_3_inexact::Vertex_const_handle v = hh->vertex();
					_triangles.insert(_triangles.begin(), poca::core::Vec3mf(CGAL::to_double(v->point().x()), CGAL::to_double(v->point().y()), CGAL::to_double(v->point().z())));
				} while (++hfc != fi->facet_begin());
			}
		}
		else
			std::cout << "ERROR !!!!!!!!!!!!!!!!" << std::endl;
#endif
	}

	void ObjectListFactory::computePoissonSurfaceObjectOMesh(const float* _xs, const float* _ys, const float* _zs, const std::set <uint32_t>& _locs, const std::vector <uint32_t>& _trianglesIndexes, const std::vector <poca::core::Vec3mf>& _normals, float& _volume, std::vector < Surface_mesh_3_double>& _meshes)
	{
		auto maxIndex = *std::max_element(_locs.begin(), _locs.end());
		std::vector <poca::core::Vec3mf> normalPerLoc;

		std::cout << __LINE__ << " - " << _locs.size() << std::endl;

		computeNormalOfLocsObject(_locs, _trianglesIndexes, _normals, normalPerLoc);
		std::cout << __LINE__ << std::endl;
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
		PointList points;
		size_t cpt = 0;
		for (auto it = _locs.begin(); it != _locs.end(); it++, cpt++) {
			auto id = *it;
			points.push_back(std::make_pair(Point_3_inexact(_xs[id], _ys[id], _zs[id]), Vector_3_inexact((double)normalPerLoc[cpt].x(), (double)normalPerLoc[cpt].y(), (double)normalPerLoc[cpt].z())));
		}
		std::cout << "\n\n### Parallel mode ###" << std::endl;
		Polyhedron poly;
		poisson_reconstruction<CGAL::Parallel_tag>(points, poly);
		_meshes.push_back(Surface_mesh_3_double());
		CGAL::copy_face_graph(poly, _meshes.back());
#else
		std::vector<pointWnormal_3_inexact> points;
		size_t cpt = 0;
		for (auto it = _locs.begin(); it != _locs.end(); it++, cpt++) {
			auto id = *it;
			points.push_back(std::make_pair(Point_3_inexact(_xs[id], _ys[id], _zs[id]), Vector_3_inexact((double)normalPerLoc[cpt].x(), (double)normalPerLoc[cpt].y(), (double)normalPerLoc[cpt].z())));
		}

		Polyhedron_3_inexact poly;
		double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(points, 6, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<pointWnormal_3_inexact>()));
		//average_spacing /= 3;

		if (CGAL::poisson_surface_reconstruction_delaunay(points.begin(), points.end(), CGAL::First_of_pair_property_map<pointWnormal_3_inexact>(), CGAL::Second_of_pair_property_map<pointWnormal_3_inexact>(), poly, average_spacing))
		{
			_meshes.push_back(Surface_mesh_3_double());
			CGAL::copy_face_graph(poly, _meshes.back());
		}
		else
			std::cout << "ERROR !!!!!!!!!!!!!!!!" << std::endl;
#endif
	}

	void ObjectListFactory::computeAlphaShape(const float* _xs, const float* _ys, const float* _zs, const std::set <uint32_t>& _locs, std::vector <poca::core::Vec3mf>& _triangles, float& _volume)
	{
		Alpha_Delaunay dt;
		for (auto id : _locs)
			dt.insert(Point_3_inexact(_xs[id], _ys[id], _zs[id]));

		std::cout << "Delaunay computed." << std::endl;
		// compute alpha shape
		Alpha_shape_3 as(dt);
		std::cout << "Alpha shape computed in REGULARIZED mode by defaut." << std::endl;
		// find optimal alpha values
		Alpha_shape_3::NT alpha_solid = as.find_alpha_solid();
		Alpha_iterator opt = as.find_optimal_alpha(1);
		std::cout << "Smallest alpha value to get a solid through data points is " << alpha_solid << std::endl;
		std::cout << "Optimal alpha value to get one connected component is " << *opt << std::endl;
		as.set_alpha(*opt);

		std::vector<Alpha_Facet> facets;
		as.get_alpha_shape_facets(std::back_inserter(facets), Alpha_shape_3::REGULAR);
		as.get_alpha_shape_facets(std::back_inserter(facets), Alpha_shape_3::SINGULAR);

		_triangles.clear();
		std::size_t nbf = facets.size();
		for (std::size_t i = 0; i < nbf; ++i)
		{
			//To have a consistent orientation of the facet, always consider an exterior cell
			if (as.classify(facets[i].first) != Alpha_shape_3::EXTERIOR)
				facets[i] = as.mirror_facet(facets[i]);
			CGAL_assertion(as.classify(facets[i].first) == Alpha_shape_3::EXTERIOR);

			int indices[3] = {
			  (facets[i].second + 1) % 4,
			  (facets[i].second + 2) % 4,
			  (facets[i].second + 3) % 4,
			};

			/// according to the encoding of vertex indices, this is needed to get
			/// a consistent orienation
			if (facets[i].second % 2 == 0) std::swap(indices[0], indices[1]);

			Alpha_Vertex_handle vs[3] = { facets[i].first->vertex(indices[2]) , facets[i].first->vertex(indices[1]) , facets[i].first->vertex(indices[0]) };
			for(auto v :vs)
				_triangles.push_back(poca::core::Vec3mf(v->point().x(), v->point().y(), v->point().z()));
		}
	}

	void ObjectListFactory::computeNormalOfLocsObject(const std::set <uint32_t>& _locs, const std::vector <uint32_t>& _trianglesIndexes, const std::vector <poca::core::Vec3mf>& _normalTriangles, std::vector <poca::core::Vec3mf>& _normalLocs)
	{
		auto maxIndex = *std::max_element(_locs.begin(), _locs.end());
		std::vector <poca::core::Vec3mf> normalPerLoc(maxIndex + 1, poca::core::Vec3mf(0.f, 0.f, 0.f));
		std::vector <float> nbPerLoc(maxIndex + 1, 0);

		for (auto id : _trianglesIndexes)
			nbPerLoc[id] += 1.f;

		for (size_t nt = 0, nn = 0; nt < _trianglesIndexes.size(); nt += 3, nn++) {
			uint32_t ids[3] = { _trianglesIndexes[nt], _trianglesIndexes[nt + 1], _trianglesIndexes[nt + 2] };
			for (auto id : ids)
				normalPerLoc[id] += _normalTriangles[nn] / nbPerLoc[id];
		}

		_normalLocs.clear();
		for (auto id : _locs)
			_normalLocs.push_back(normalPerLoc[id]);
	}
}

