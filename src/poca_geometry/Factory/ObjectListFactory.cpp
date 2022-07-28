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

#include <General/BasicComponent.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/ROIInterface.hpp>
#include <DesignPatterns/GlobalParametersSingleton.hpp>
#include <General/Misc.h>

#include "ObjectListFactory.hpp"
#include "../Interfaces/DelaunayTriangulationInterface.hpp"
#include "../Interfaces/DelaunayTriangulationFactoryInterface.hpp"
#include "../Geometry/DelaunayTriangulation.hpp"
#include "../Geometry/BasicComputation.hpp"
#include "../Geometry/ObjectList.hpp"
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

	ObjectList* ObjectListFactory::createObjectList(poca::core::MyObjectInterface* _obj, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const bool _inROIs)
	{
		poca::core::BasicComponent* bci = _obj->getBasicComponent("DelaunayTriangulation");
		DelaunayTriangulationInterface* delaunay = dynamic_cast <DelaunayTriangulationInterface*>(bci);
		if (!delaunay) return NULL;
		const std::vector <poca::core::ROIInterface*>& ROIs = _inROIs ? _obj->getROIs() : std::vector <poca::core::ROIInterface*>();
		return createObjectList(delaunay, _selection, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea, ROIs);
	}
	ObjectList* ObjectListFactory::createObjectList(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		std::vector <bool> selectionDelaunay;
		_delaunay->generateFaceSelectionFromLocSelection(_selection, selectionDelaunay);
		return createObjectListFromDelaunay(_delaunay, selectionDelaunay, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea, _ROIs);
	}

	ObjectList* ObjectListFactory::createObjectListFromDelaunay(poca::core::MyObjectInterface* _obj, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const bool _inROIs)
	{
		poca::core::BasicComponent* bci = _obj->getBasicComponent("DelaunayTriangulation");
		DelaunayTriangulationInterface* delaunay = dynamic_cast <DelaunayTriangulationInterface*>(bci);
		if (!delaunay) return NULL;
		const std::vector <poca::core::ROIInterface*>& ROIs = _inROIs ? _obj->getROIs() : std::vector <poca::core::ROIInterface*>();
		return createObjectListFromDelaunay(delaunay, _selection, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea, ROIs);
	}

	ObjectList* ObjectListFactory::createObjectListFromDelaunay(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		clock_t t1, t2;
		t1 = clock();
		ObjectList* objs = NULL;
		if (_delaunay->dimension() == 2)
			objs = createObjectList2D(_delaunay, _selection, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea, _ROIs);
		else if (_delaunay->dimension() == 3)
			objs = createObjectList3D(_delaunay, _selection, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea, _ROIs);
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("Time for creating objects: %ld ms\n", elapsed);
		return objs;
	}

	ObjectList* ObjectListFactory::createObjectListAlreadyIdentified(poca::core::MyObjectInterface* _obj, const std::vector <uint32_t>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea)
	{
		poca::core::BasicComponent* bci = _obj->getBasicComponent("DelaunayTriangulation");
		DelaunayTriangulationInterface* delaunay = dynamic_cast <DelaunayTriangulationInterface*>(bci);
		if (!delaunay) return NULL;
		return createObjectListAlreadyIdentified(delaunay, _selection, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea);
	}

	ObjectList* ObjectListFactory::createObjectListAlreadyIdentified(DelaunayTriangulationInterface* _delaunay, const std::vector <uint32_t>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea)
	{
		clock_t t1, t2;
		t1 = clock();
		ObjectList* objs = NULL;
		std::map <uint32_t, std::vector <uint32_t>> selectionDelaunay;
		_delaunay->generateFaceSelectionFromLocSelection(_selection, selectionDelaunay);
		if (_delaunay->dimension() == 2)
			objs = createObjectList2D(_delaunay, selectionDelaunay, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea);
		else if (_delaunay->dimension() == 3)
			objs = createObjectList3D(_delaunay, selectionDelaunay, _dMax, _minNbLocs, _maxNbLocs, _minArea, _maxArea);
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("Time for creating objects: %ld ms\n", elapsed);
		return objs;
	}

	ObjectList* ObjectListFactory::createObjectList2D(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
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
		float area = 0.f;
		for (uint32_t n = 0; n < _delaunay->nbFaces(); n++) {
			if (!selectionTriangulationFaces[n]) continue;
			std::vector <uint32_t> queueTriangles;
			std::set <uint32_t> locsOfObject;
			std::vector <poca::core::Vec3mf> trianglesOfObject, outlineOfObject;
			queueTriangles.push_back(n);
			size_t currentTriangle = 0, sizeQueue = queueTriangles.size();

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
		return locsAllObjects.empty() ? NULL : new ObjectList(xs, ys, _delaunay->getZs() == NULL ? NULL : _delaunay->getZs(), locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, outlinesAllObjects, firstOutlines, linkTriangulationFacesToObjects);
	}

	ObjectList* ObjectListFactory::createObjectList3D(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		const float* zs = _delaunay->getZs();
		const std::vector <float>& volumes = _delaunay->getOriginalHistogram("volume")->getValues();
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
				ObjectListFactoryInterface::TypeShape type = poca::core::GlobalParametersSingleton::instance()->m_parameters["typeObject"].get<ObjectListFactoryInterface::TypeShape>();
				switch (type) {
				case ObjectListFactoryInterface::TRIANGULATION:
					for (const auto id : indexTrianglesOfObject)
						trianglesOfObject.push_back(poca::core::Vec3mf(xs[id], ys[id], zs[id]));
					break;
				case ObjectListFactoryInterface::CONVEX_HULL:
					computeConvexHullObject(xs, ys, zs, locsOfOutline, trianglesOfObject, volume);
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
		return objs;
	}

	ObjectList* ObjectListFactory::createObjectList2D(DelaunayTriangulationInterface* _delaunay, const std::vector <uint32_t>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
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
	}

	ObjectList* ObjectListFactory::createObjectList2D(DelaunayTriangulationInterface* _delaunay, const std::map <uint32_t, std::vector <uint32_t>>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea)
	{
		const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		std::vector <float> zsTmp;
		if (_delaunay->getZs() == NULL)
			zsTmp = std::vector<float>(_delaunay->nbPoints(), 0.f);
		const float* zs = _delaunay->getZs() == NULL ? zsTmp.data() : _delaunay->getZs();

		const std::vector<uint32_t>& triangles = _delaunay->getTriangles();
		const poca::core::MyArrayUInt32& neighbors = _delaunay->getNeighbors();

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
			std::set <uint32_t> locsOfObject;
			std::vector <poca::core::Vec3mf> trianglesOfObject, outlineOfObject;
			for (auto index : it->second) {
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
					if (globalSelection[indexNeigh] != globalSelection[index]){
						std::array<size_t, 3> edge = _delaunay->getOutline(index, i);
						outlineOfObject.push_back(poca::core::Vec3mf(xs[edge[0]], ys[edge[0]], zs[edge[0]]));
						outlineOfObject.push_back(poca::core::Vec3mf(xs[edge[1]], ys[edge[1]], zs[edge[1]]));
					}
				}
			}
			if (_minNbLocs <= locsOfObject.size() && locsOfObject.size() <= _maxNbLocs && _minArea <= area && area <= _maxArea) {
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
		return locsAllObjects.empty() ? NULL : new ObjectList(xs, ys, _delaunay->getZs() == NULL ? NULL : _delaunay->getZs(), locsAllObjects, firstsLocs, trianglesAllObjects, firstTriangles, outlinesAllObjects, firstOutlines, globalSelection);
	}

	ObjectList* ObjectListFactory::createObjectList3D(DelaunayTriangulationInterface* _delaunay, const std::vector <uint32_t>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea, const std::vector <poca::core::ROIInterface*>& _ROIs)
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
	}

	ObjectList* ObjectListFactory::createObjectList3D(DelaunayTriangulationInterface* _delaunay, const std::map <uint32_t, std::vector <uint32_t>>& _selection, const float _dMax, const size_t _minNbLocs, const size_t _maxNbLocs, const float _minArea, const float _maxArea)
	{
		const float* xs = _delaunay->getXs();
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
				ObjectListFactoryInterface::TypeShape type = poca::core::GlobalParametersSingleton::instance()->m_parameters["typeObject"].get<ObjectListFactoryInterface::TypeShape>();
				switch (type) {
				case ObjectListFactoryInterface::TRIANGULATION:
					for (const auto id : indexTrianglesOfObject)
						trianglesOfObject.push_back(poca::core::Vec3mf(xs[id], ys[id], zs[id]));
					break;
				case ObjectListFactoryInterface::CONVEX_HULL:
					computeConvexHullObject(xs, ys, zs, locsOfOutline, trianglesOfObject, volume);
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
		return objs;
	}

	void ObjectListFactory::computeConvexHullObject(const float* _xs, const float* _ys, const float* _zs, const std::set <uint32_t>& _locs, std::vector <poca::core::Vec3mf>& _triangles, float& _volume)
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
		_volume = CGAL::Polygon_mesh_processing::volume(poly);
	}

	void ObjectListFactory::computePoissonSurfaceObject(const float* _xs, const float* _ys, const float* _zs, const std::set <uint32_t>& _locs, const std::vector <uint32_t>& _trianglesIndexes, const std::vector <poca::core::Vec3mf>& _normals, std::vector <poca::core::Vec3mf>& _triangles, float& _volume)
	{
		auto maxIndex = *std::max_element(_locs.begin(), _locs.end());
		std::vector <poca::core::Vec3mf> normalPerLoc;
		
		computeNormalOfLocsObject(_locs, _trianglesIndexes, _normals, normalPerLoc);

		std::vector<pointWnormal_3_inexact> points;
		size_t cpt = 0;
		for (auto it = _locs.begin(); it != _locs.end(); it++, cpt++) {
			auto id = *it;
			points.push_back(std::make_pair(Point_3_inexact(_xs[id], _ys[id], _zs[id]), Vector_3_inexact((double)normalPerLoc[cpt].x(), (double)normalPerLoc[cpt].y(), (double)normalPerLoc[cpt].z())));
		}

		Polyhedron_3_inexact poly;
		double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(points, 6, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<pointWnormal_3_inexact>()));
		average_spacing /= 3;

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

