/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListFactory.cpp
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

#include <fstream>

#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/Alpha_shape_3.h>
#include <CGAL/Alpha_shape_cell_base_3.h>
#include <CGAL/Alpha_shape_vertex_base_3.h>
#include <CGAL/convex_hull_2.h>
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
#include <CGAL/Point_set_3.h>
#include <CGAL/Poisson_reconstruction_function.h>
#include <CGAL/Implicit_surface_3.h>
#endif

#include <General/BasicComponent.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/ROIInterface.hpp>
#include <General/Histogram.hpp>
#include <General/Engine.hpp>
#include <General/Misc.h>

#include "ObjectListFactory.hpp"
#include "../Interfaces/DelaunayTriangulationInterface.hpp"
#include "../Interfaces/DelaunayTriangulationFactoryInterface.hpp"
#include "../Geometry/DelaunayTriangulation.hpp"
#include "../Geometry/BasicComputation.hpp"
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

#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
typedef CGAL::Point_set_3<Point_3_inexact, Vector_3_inexact> Point_set;
#endif

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
		if (!delaunay) return NULL;
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
		if (!delaunay) return NULL;
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

			float area = 0.f;
			while (currentTriangle < sizeQueue) {
				size_t index = queueTriangles.at(currentTriangle);
				if (selectionTriangulationFaces[index]) {
					selectionTriangulationFaces[index] = false;
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
						if (indexNeigh == std::numeric_limits<std::uint32_t>::max() || !originalSelection[indexNeigh]) {
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
			}
		}
		ObjectListInterface* objs = locsAllObjects.empty() ? NULL : new ObjectListDelaunay(xs, ys, zs, locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, volumeObjects, linkTriangulationFacesToObjects, locsAllOutlines, firstOutlineLocs, normalsAllOutlineLocs);
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
					computeConvexHullObject3D(xs, ys, zs, locs, outlinesAllObjects, triCHull, area);
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
		firstsLocs.push_back(locsAllObjects.size());
		firstTriangles.push_back(trianglesAllObjects.size());
		firstOutlines.push_back(outlinesAllObjects.size());

		//return locsAllObjects.empty() ? NULL : new ObjectList(xs, ys, zs, locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, outlinesAllObjects, firstOutlines, std::vector <uint32_t>());
		return locsAllObjects.empty() ? NULL : new ObjectListDelaunay(xs, ys, zs, locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, volumeObjects, std::vector<uint32_t>(), locsAllObjects, firstsLocs, std::vector <poca::core::Vec3mf>());

		/*const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		const float* zs = _delaunay->getZs();
		const std::vector <float>& volumes = _delaunay->getOriginalHistogram("volume")->getValues();
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
		_delaunay->executeCommand(false, "updateFeature");

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
				
			}
		}

		ObjectList* objs = locsAllObjects.empty() ? NULL : new ObjectList(xs, ys, zs, locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, volumeObjects, linkTriangulationFacesToObjects, locsAllOutlines, firstOutlineLocs, normalsAllOutlineLocs);
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
		FT sm_angle = 20.0; // Min triangle angle in degrees.
		FT sm_radius = 100; // Max triangle size w.r.t. point set average spacing.
		FT sm_distance = 0.25; // Surface Approximation error w.r.t. point set average spacing.

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
		average_spacing /= 3;

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

