/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectColocalization.cpp
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

#include <fstream>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>

#include <Geometry/DetectionSet.hpp>
#include <Geometry/CGAL_includes.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/DelaunayTriangulationFactoryInterface.hpp>
#include <Interfaces/DelaunayTriangulationInterface.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <General/Misc.h>
#include <General/MyArray.hpp>
#include <General/MyData.hpp>

#include "ObjectColocalization.hpp"


ObjectColocalization::ObjectColocalization(poca::core::MyObjectInterface* _obj1, poca::core::MyObjectInterface* _obj2, const bool _sampling, const float _d, const uint32_t _sub, const uint32_t _minNbLocs) : poca::core::BasicComponent("ObjectColocalization"), m_samplingEnabled(_sampling), m_distance2D(_d), m_subdiv3D(_sub), m_minNbLocs(_minNbLocs), m_objectList(NULL), m_delaunay(NULL)
{
	m_objects[0] = _obj1; m_objects[1] = _obj2;

	poca::geometry::ObjectListDelaunay* objectsDelaunay[] = { static_cast<poca::geometry::ObjectListDelaunay*>(m_objectsPerColor[0]->currentObjectList()),  static_cast<poca::geometry::ObjectListDelaunay*>(m_objectsPerColor[1]->currentObjectList()) };
	if (objectsDelaunay[0] == NULL || objectsDelaunay[1] == NULL)
		return;

	computeOverlapLast();
	//Compute stats
	std::vector <std::vector<uint32_t>> infoColor1, infoColor2;

	infoColor1.resize(m_nbObjectsPerColor[0]);
	infoColor2.resize(m_nbObjectsPerColor[1]);

	if (m_objectList == NULL) return;

	std::cout << "Begin link overlap -> orig objets" << std::endl;
	const uint32_t INVALID = std::numeric_limits<std::uint32_t>::max();
	const poca::core::MyArrayUInt32& locs = m_objectList->getLocsObjects();
	for (size_t i = 0; i < locs.nbElements(); i++) {
		uint32_t idxColor1 = INVALID, idxColor2 = INVALID;
		uint32_t vertex = locs.elementIObject(i, 0);
		idxColor1 = m_linkLocsToObjectsBothColor2[vertex][0];
		idxColor2 = m_linkLocsToObjectsBothColor2[vertex][1];
		if (idxColor1 == INVALID || idxColor2 == INVALID)
			std::cout << "Problem with finding links between overlap and original objects for coloc object " << i << "[" << idxColor1 << ", " << idxColor2 << "]" << std::endl;
		else {
			if (idxColor1 != INVALID) infoColor1[idxColor1].push_back(idxColor2);
			if (idxColor2 != INVALID) infoColor2[idxColor2].push_back(idxColor1);
		}
		m_infoColoc.push_back(std::make_pair(idxColor1, idxColor2));
		///or (size_t j = 0; j < locs.nbElementsObject(i) && (idxColor1 == INVALID || idxColor2 == INVALID); j++) {
		//	uint32_t vertex = locs.elementIObject(i, j);
		//	if (m_linkLocsToObjectsBothColor[vertex] < m_nbObjectsPerColor[0])
		//		idxColor1 = m_linkLocsToObjectsBothColor[vertex];
		//	else
		//		idxColor2 = m_linkLocsToObjectsBothColor[vertex] - m_nbObjectsPerColor[0];
		//}
		//if (idxColor1 == INVALID || idxColor2 == INVALID)
		//	std::cout << "Problem with finding links between overlap and original objects for coloc object " << i << "[" << idxColor1 << ", " << idxColor2 << "]" << std::endl;
		//else {
		//	if(idxColor1 != INVALID) infoColor1[idxColor1].push_back(idxColor2);
		//	if(idxColor2 != INVALID) infoColor2[idxColor2].push_back(idxColor1);
		//}
		//m_infoColoc.push_back(std::make_pair(idxColor1, idxColor2));
	}
	m_infoColor1.initialize(infoColor1);
	m_infoColor2.initialize(infoColor2);

	//Now we compute the locs that are part of the coloclized part of the objects
	//This is required since we added point to compute a better approximation of the overlap volume
	//These added points must not be counted to have a proper information about the locs part of the overlap
	int idx[] = { 0, 1 };

	poca::geometry::KdTree_DetectionPoint* kdtrees[2] = { NULL, NULL };
	for (auto n : idx) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("DetectionSet");
		poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
		if (dset) {
			kdtrees[n] = dset->getKdTree();
		}
	}
	poca::geometry::DelaunayTriangulationInterface* delaunays[2] = { NULL, NULL };
	for (auto n : idx) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("DelaunayTriangulation");
		poca::geometry::DelaunayTriangulationInterface* del = dynamic_cast <poca::geometry::DelaunayTriangulationInterface*>(bci);
		if (del)
			delaunays[n] = del;
	}
	std::vector <poca::core::Vec3mf> locsPerOverlap;
	std::vector <uint32_t> idxLocsObjOverlap, rangeLocsObjOverlap;
	rangeLocsObjOverlap.push_back(0);
	uint32_t idColor[] = { 0, 1 };
	uint32_t idOtherColor[] = { 1, 0 };
	const std::size_t num_results = 1;
	std::vector<size_t> ret_index(num_results);
	std::vector<double> out_dist_sqr(num_results);
	std::vector <uint32_t> selectionDelaunayElements[2] = { objectsDelaunay[0]->getLinkTriangulationFacesToObjects(), objectsDelaunay[1]->getLinkTriangulationFacesToObjects() };
	for (size_t n = 0; n < m_infoColoc.size(); n++) {
		uint32_t idObjColor[] = { m_infoColoc.at(n).first, m_infoColoc.at(n).second };

		for (auto cur = 0; cur < 2; cur++) {
			auto curColor = idColor[cur], otherColor = idOtherColor[cur];
			const poca::core::MyArrayUInt32& locs = m_objectsPerColor[curColor]->currentObjectList()->getLocsObjects();
			const std::vector <uint32_t>& rangeLocs = locs.getFirstElements();
			const std::vector <uint32_t>& idxLocs = locs.getData();
			const float* xsLocs = objectsDelaunay[curColor]->getXs();
			const float* ysLocs = objectsDelaunay[curColor]->getYs();
			const float* zsLosc = objectsDelaunay[curColor]->getZs();

			poca::geometry::KdTree_DetectionPoint* otherKdtree = kdtrees[otherColor];
			poca::geometry::DelaunayTriangulationInterface* otherD = delaunays[otherColor];

			uint32_t idObj = idObjColor[curColor], idOtherObj = idObjColor[otherColor];

			for (uint32_t j = rangeLocs[idObj]; j < rangeLocs[idObj + 1]; j++) {
				uint32_t idLoc = idxLocs[j];
				float x = xsLocs[idLoc], y = ysLocs[idLoc], z = zsLosc != NULL ? zsLosc[idLoc] : 0;
				const double queryPt[3] = { x, y, z };
				otherKdtree->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
				uint32_t idxDelaunayFace = otherD->indexTriangleOfPoint(x, y, z, ret_index[0]);
				if (idxDelaunayFace != std::numeric_limits<std::uint32_t>::max() && selectionDelaunayElements[otherColor][idxDelaunayFace] == idOtherObj) {
					idxLocsObjOverlap.push_back(m_xsOrigPoints.size());
					locsPerOverlap.push_back(poca::core::Vec3mf(x, y, z));
					m_xsOrigPoints.push_back(x);
					m_ysOrigPoints.push_back(y);
					if(zsLosc != NULL)
						m_zsOrigPoints.push_back(z);
				}
			}
		}
		rangeLocsObjOverlap.push_back(locsPerOverlap.size());
	}
	m_origPointsInOverlapObjects.initialize(locsPerOverlap, rangeLocsObjOverlap);
	m_objectList->setLocs(m_xsOrigPoints.data(), m_ysOrigPoints.data(), m_zsOrigPoints.empty() ? NULL : m_zsOrigPoints.data(), idxLocsObjOverlap, rangeLocsObjOverlap);
	std::cout << "End link overlap -> orig objets" << std::endl;
}

ObjectColocalization::~ObjectColocalization()
{
}

poca::core::BasicComponentInterface* ObjectColocalization::copy()
{
	return new ObjectColocalization(*this);
}

void ObjectColocalization::computeOverlapLast2()
{
	poca::geometry::KdTree_DetectionPoint* kdtrees[2] = { NULL, NULL };
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("DetectionSet");
		poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
		if (dset) {
			kdtrees[n] = dset->getKdTree();
		}
	}
	poca::geometry::DelaunayTriangulationInterface* delaunays[2] = { NULL, NULL };
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("DelaunayTriangulation");
		poca::geometry::DelaunayTriangulationInterface* del = dynamic_cast <poca::geometry::DelaunayTriangulationInterface*>(bci);
		if (del)
			delaunays[n] = del;
	}
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("ObjectLists");
		poca::geometry::ObjectLists* obj = dynamic_cast <poca::geometry::ObjectLists*>(bci);
		if (obj)
			m_objectsPerColor[n] = obj;
	}
	m_dimension = m_objectsPerColor[0]->currentObjectList()->dimension();

	poca::geometry::ObjectListDelaunay* objectsDelaunay[] = { static_cast<poca::geometry::ObjectListDelaunay*>(m_objectsPerColor[0]->currentObjectList()),  static_cast<poca::geometry::ObjectListDelaunay*>(m_objectsPerColor[1]->currentObjectList()) };
	if (objectsDelaunay[0] == NULL || objectsDelaunay[1] == NULL)
		return;

	std::vector <uint32_t> selectionDelaunayElements[2] = { objectsDelaunay[0]->getLinkTriangulationFacesToObjects(), objectsDelaunay[1]->getLinkTriangulationFacesToObjects() };
	size_t nbLocsObjs[2], firstIndexObjPerColor = 0;

	const std::size_t num_results = 1;
	std::vector<size_t> ret_index(num_results);
	std::vector<double> out_dist_sqr(num_results);

	for (size_t n = 0; n < 2; n++) {
		const float* xsobj = objectsDelaunay[n]->getXs(), * ysobj = objectsDelaunay[n]->getYs(), * zsobj = objectsDelaunay[n]->getZs();
		const poca::core::MyArrayUInt32& locs = m_objectsPerColor[n]->currentObjectList()->getLocsObjects();
		const std::vector <uint32_t>& data = locs.getData();
		const std::vector <uint32_t>& indexes = locs.getFirstElements();

		uint32_t otherId = (n + 1) % 2;
		poca::geometry::DelaunayTriangulationInterface* otherD = delaunays[otherId];
		poca::geometry::KdTree_DetectionPoint* otherKdtree = kdtrees[otherId];

		m_nbObjectsPerColor[n] = locs.nbElements();
		if (m_dimension == 2) {
			//Add some points on the objects contour, to properly sample them
			const poca::core::MyArrayVec3mf& outlines = m_objectsPerColor[n]->currentObjectList()->getOutlinesObjects();
			const std::vector <poca::core::Vec3mf>& dataO = outlines.getData();
			const std::vector <uint32_t>& indexesO = outlines.getFirstElements();
			for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
				for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j++) {
					const poca::core::Vec3mf& p = dataO[j];
					m_xs.push_back(p.x());
					m_ys.push_back(p.y());
					if (zsobj != NULL) m_zs.push_back(p.z());
					m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
				}
			if (m_samplingEnabled) {
				for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
					for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j++) {
						for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j += 2) {
							//for (size_t i = 0; i < dataO.size(); i += 2) {
							const poca::core::Vec3mf& p1 = dataO[j], & p2 = dataO[j + 1];
							poca::core::Vec3mf v = p2 - p1;
							float dvector = v.length(), d = m_distance2D;
							v.normalize();
							for (; d < dvector; d += m_distance2D) {
								poca::core::Vec3mf p = p1 + v * d;
								m_xs.push_back(p.x());
								m_ys.push_back(p.y());
								if (zsobj != NULL) m_zs.push_back(p.z());
								m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
							}
						}
					}
			}
		}
		else {
			//Add some points on the objects surface, to properly sample them
			const poca::core::MyArrayVec3mf& triangles = m_objectsPerColor[n]->currentObjectList()->getTrianglesObjects();
			const std::vector <poca::core::Vec3mf>& dataO = triangles.getData();
			const std::vector <uint32_t>& indexesO = triangles.getFirstElements();
			std::vector <bool> cutTriangles;

			std::vector <bool> insides = { false, false, false };
			bool isIn = false, isOut = false;

			double nbPs = m_samplingEnabled ? 2 * m_nbObjectsPerColor[n] : m_nbObjectsPerColor[n];
			unsigned int nbForUpdate = nbPs / 100.;
			if (nbForUpdate == 0) nbForUpdate = 1;

			//Only contours of the objects
			size_t cpt = 0;
			for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++, cpt++) {
				if (cpt % nbForUpdate == 0) printf("\rDetermination of coloc locs for color %i: %.2f %%", (n + 1), ((double)cpt / nbPs * 100.));
				std::set <poca::core::Vec3mf> objPoints;
				for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j += 3) {
					std::vector <uint32_t> indexes = { j, j + 1, j + 2 };

					isIn = false; isOut = false;

					for (size_t idx = 0; idx < indexes.size(); idx++) {
						uint32_t index = indexes[idx];
						float x = dataO[index].x(), y = dataO[index].y(), z = dataO[index].z();
						const double queryPt[3] = { x, y, z };
						otherKdtree->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
						uint32_t idxDelaunayFace = otherD->indexTriangleOfPoint(x, y, z, ret_index[0]);
						insides[idx] = idxDelaunayFace != std::numeric_limits<std::uint32_t>::max() && selectionDelaunayElements[otherId][idxDelaunayFace] != std::numeric_limits<std::uint32_t>::max();
						isIn |= insides[idx];
						isOut |= !insides[idx];
						if (insides[idx])
							objPoints.insert(dataO[index]);
					}
					cutTriangles.push_back(isIn && isOut);
				}
				for (const auto& p : objPoints) {
					m_xs.push_back(p.x());
					m_ys.push_back(p.y());
					m_zs.push_back(p.z());
					m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);

				}
			}
			if (m_samplingEnabled) {
				size_t cptTrianglesCutOrNot = 0;
				for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++, cpt++) {
					if (cpt % nbForUpdate == 0) printf("\rDetermination of coloc locs for color %i: %.2f %%", n + 1, ((double)cpt / nbPs * 100.));
					for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j += 3) {
						if (!cutTriangles[cptTrianglesCutOrNot++]) continue;

						std::vector <poca::core::Vec3mf> queueTriangles = { dataO[j], dataO[j + 1], dataO[j + 2] };
						size_t first = 0, last = queueTriangles.size();
						for (uint32_t k = 0; k < m_subdiv3D; k++) {
							for (size_t cur = first; cur < last; cur += 3) {
								const poca::core::Vec3mf p1 = queueTriangles[cur], p2 = queueTriangles[cur + 1], p3 = queueTriangles[cur + 2], centroid = (p1 + p2 + p3) / 3.f;

								const double queryPt[3] = { centroid.x(), centroid.y(), centroid.z() };
								otherKdtree->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
								uint32_t idxDelaunayFace = otherD->indexTriangleOfPoint(centroid.x(), centroid.y(), centroid.z(), ret_index[0]);
								if (idxDelaunayFace != std::numeric_limits<std::uint32_t>::max() && selectionDelaunayElements[otherId][idxDelaunayFace] != std::numeric_limits<std::uint32_t>::max()) {
									m_xs.push_back(centroid.x());
									m_ys.push_back(centroid.y());
									m_zs.push_back(centroid.z());
									m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
								}

								//Add the 3 new triangles
								queueTriangles.push_back(p1);
								queueTriangles.push_back(p2);
								queueTriangles.push_back(centroid);
								queueTriangles.push_back(p2);
								queueTriangles.push_back(p3);
								queueTriangles.push_back(centroid);
								queueTriangles.push_back(p3);
								queueTriangles.push_back(p1);
								queueTriangles.push_back(centroid);
							}
							first = last;
							last = queueTriangles.size();
						}
					}
				}
			}
		}
		nbLocsObjs[n] = m_xs.size();
		firstIndexObjPerColor += m_nbObjectsPerColor[n];
		printf("\rDetermination of coloc locs for color 100.00 %%", n + 1);
	}

	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	poca::geometry::DelaunayTriangulationFactoryInterface* factory = poca::geometry::createDelaunayTriangulationFactory();
	m_delaunay = m_zs.empty() ? factory->createDelaunayTriangulation(m_xs, m_ys) : factory->createDelaunayTriangulation(m_xs, m_ys, m_zs);
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;

	m_selectedTetrahedra.resize(m_delaunay->nbFaces());
	std::fill(m_selectedTetrahedra.begin(), m_selectedTetrahedra.end(), true);

	//Now, we need to determine which tetrahedra of the delaunay are part of coloc objects
	//To do so we compute the centroid of the tetrahedra and test if it belongs to objects of the two colors
	const std::vector<uint32_t>& triangles = m_delaunay->getTriangles();
	const poca::core::MyArrayUInt32& neighbors = m_delaunay->getNeighbors();
	const std::vector <uint32_t> indiceTriangles = neighbors.getFirstElements();
	const float* xs = m_delaunay->getXs();
	const float* ys = m_delaunay->getYs();
	const float* zs = m_delaunay->getZs();
	if (m_dimension == 2) {
		for (uint32_t n = 0, cpt = 0; n < triangles.size(); n += 3, cpt++) {
			size_t indexes[3] = { triangles[n], triangles[n + 1], triangles[n + 2] };
			poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
			for (size_t index : indexes)
				centroid += (poca::core::Vec3mf(xs[index], ys[index], zs[index]) / 3.f);
			bool insideBoth = true;
			for (size_t i = 0; i < 2; i++) {
				const double queryPt[3] = { centroid.x(), centroid.y(), centroid.z() };
				kdtrees[i]->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
				uint32_t idxDelaunayFace = delaunays[i]->indexTriangleOfPoint(centroid.x(), centroid.y(), centroid.z(), ret_index[0]);
				insideBoth &= idxDelaunayFace != std::numeric_limits<std::uint32_t>::max() && selectionDelaunayElements[i][idxDelaunayFace] != std::numeric_limits<std::uint32_t>::max();
			}
			m_selectedTetrahedra[cpt] = insideBoth;
		}
	}
	else {
		for (uint32_t n = 0, cpt = 0; n < triangles.size(); n += 12, cpt++) {
			size_t indexes[4] = { triangles[n], triangles[n + 3 * 1], triangles[n + 3 * 2], triangles[n + 3 * 3] };
			poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
			for (size_t index : indexes)
				centroid += (poca::core::Vec3mf(xs[index], ys[index], zs[index]) / 4.f);
			bool insideBoth = true;
			for (size_t i = 0; i < 2; i++) {
				//const double queryPt[3] = { centroid.x(), centroid.y(), centroid.z() };
				//kdtrees[i]->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
				uint32_t idxDelaunayFace = delaunays[i]->indexTriangleOfPoint(centroid.x(), centroid.y(), centroid.z());
				bool res = idxDelaunayFace != std::numeric_limits<std::uint32_t>::max() && selectionDelaunayElements[i][idxDelaunayFace] != std::numeric_limits<std::uint32_t>::max();
				insideBoth &= res;
			}
			m_selectedTetrahedra[cpt] = insideBoth;
		}
	}

	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	poca::geometry::ObjectListFactory factoryObjects;
	//m_objectList = factoryObjects.createObjectListAlreadyIdentified(m_delaunay, m_selectionIntersection, FLT_MAX, m_minNbLocs);
	m_objectList = static_cast<poca::geometry::ObjectListDelaunay*>(factoryObjects.createObjectListFromDelaunay(m_delaunay, m_selectedTetrahedra, FLT_MAX, m_minNbLocs));
	//m_objectList = factoryObjects.createObjectList(m_delaunay, selectedLocsOverlapObjs, FLT_MAX, m_minNbLocs);
	m_objectList->setBoundingBox(m_delaunay->boundingBox());

	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	m_triangles = m_objectList->getTrianglesObjects().getData();
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;


}

void ObjectColocalization::computeOverlapLast()
{
	poca::geometry::ObjectListDelaunay* objectsDelaunay[] = { static_cast<poca::geometry::ObjectListDelaunay*>(m_objectsPerColor[0]->currentObjectList()),  static_cast<poca::geometry::ObjectListDelaunay*>(m_objectsPerColor[1]->currentObjectList()) };
	if (objectsDelaunay[0] == NULL || objectsDelaunay[1] == NULL)
		return;

	poca::geometry::KdTree_DetectionPoint* kdtrees[2] = { NULL, NULL };
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("DetectionSet");
		poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
		if (dset) {
			kdtrees[n] = dset->getKdTree();
		}
	}
	poca::geometry::DelaunayTriangulationInterface* delaunays[2] = { NULL, NULL };
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("DelaunayTriangulation");
		poca::geometry::DelaunayTriangulationInterface* del = dynamic_cast <poca::geometry::DelaunayTriangulationInterface*>(bci);
		if (del)
			delaunays[n] = del;
	}
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("ObjectLists");
		poca::geometry::ObjectLists* obj = dynamic_cast <poca::geometry::ObjectLists*>(bci);
		if (obj)
			m_objectsPerColor[n] = obj;
	}
	m_dimension = m_objectsPerColor[0]->currentObjectList()->dimension();

	std::vector <uint32_t> selectionDelaunayElements[2] = { objectsDelaunay[0]->getLinkTriangulationFacesToObjects(), objectsDelaunay[1]->getLinkTriangulationFacesToObjects() };
	size_t nbLocsObjs[2], firstIndexObjPerColor = 0;

	const std::size_t num_results = 1;
	std::vector<size_t> ret_index(num_results);
	std::vector<double> out_dist_sqr(num_results);

	for (size_t n = 0; n < 2; n++) {
		const float* xsobj = objectsDelaunay[n]->getXs(), * ysobj = objectsDelaunay[n]->getYs(), * zsobj = objectsDelaunay[n]->getZs();
		const poca::core::MyArrayUInt32& locs = m_objectsPerColor[n]->currentObjectList()->getLocsObjects();
		const std::vector <uint32_t>& data = locs.getData();
		const std::vector <uint32_t>& indexes = locs.getFirstElements();
	
		uint32_t otherId = (n + 1) % 2;
		poca::geometry::DelaunayTriangulationInterface* otherD = delaunays[otherId];
		poca::geometry::KdTree_DetectionPoint* otherKdtree = kdtrees[otherId];
		
		m_nbObjectsPerColor[n] = locs.nbElements();
		if (m_dimension == 2) {
			//Add some points on the objects contour, to properly sample them
			const poca::core::MyArrayVec3mf& outlines = m_objectsPerColor[n]->currentObjectList()->getOutlinesObjects();
			const std::vector <poca::core::Vec3mf>& dataO = outlines.getData();
			const std::vector <uint32_t>& indexesO = outlines.getFirstElements();
			for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
				for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j++) {
					const poca::core::Vec3mf& p = dataO[j];
					m_xs.push_back(p.x());
					m_ys.push_back(p.y());
					if (zsobj != NULL) m_zs.push_back(p.z());
					m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
				}
			if (m_samplingEnabled) {
				for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
					for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j++) {
						for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j += 2) {
							//for (size_t i = 0; i < dataO.size(); i += 2) {
							const poca::core::Vec3mf& p1 = dataO[j], & p2 = dataO[j + 1];
							poca::core::Vec3mf v = p2 - p1;
							float dvector = v.length(), d = m_distance2D;
							v.normalize();
							for (; d < dvector; d += m_distance2D) {
								poca::core::Vec3mf p = p1 + v * d;
								m_xs.push_back(p.x());
								m_ys.push_back(p.y());
								if (zsobj != NULL) m_zs.push_back(p.z());
								m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
							}
						}
					}
			}
		}
		else {
			//Add some points on the objects surface, to properly sample them
			const poca::core::MyArrayVec3mf& triangles = m_objectsPerColor[n]->currentObjectList()->getTrianglesObjects();
			const std::vector <poca::core::Vec3mf>& dataO = triangles.getData();
			const std::vector <uint32_t>& indexesO = triangles.getFirstElements();
			std::vector <bool> cutTriangles;

			std::vector <bool> insides = { false, false, false };
			bool isIn = false, isOut = false;

			double nbPs = m_samplingEnabled ? 2 * m_nbObjectsPerColor[n] : m_nbObjectsPerColor[n];
			unsigned int nbForUpdate = nbPs / 100.;
			if (nbForUpdate == 0) nbForUpdate = 1;
			
			//Only contours of the objects
			size_t cpt = 0;
			for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++, cpt++) {
				if (cpt % nbForUpdate == 0) printf("\rDetermination of coloc locs for color %i: %.2f %%", (n + 1),  ((double)cpt / nbPs * 100.));
				std::set <poca::core::Vec3mf> objPoints;
				for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j+=3) {
					std::vector <uint32_t> indexes = { j, j + 1, j + 2 };
					
					isIn = false; isOut = false;

					for (size_t idx = 0; idx < indexes.size(); idx++) {
						uint32_t index = indexes[idx];
						float x = dataO[index].x(), y = dataO[index].y(), z = dataO[index].z();
						const double queryPt[3] = { x, y, z };
						otherKdtree->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
						uint32_t idxDelaunayFace = otherD->indexTriangleOfPoint(x, y, z, ret_index[0]);
						insides[idx] = idxDelaunayFace != std::numeric_limits<std::uint32_t>::max() && selectionDelaunayElements[otherId][idxDelaunayFace] != std::numeric_limits<std::uint32_t>::max();
						isIn |= insides[idx];
						isOut |= !insides[idx];
						if (insides[idx])
							objPoints.insert(dataO[index]);
					}
					cutTriangles.push_back(true);// isIn&& isOut);
				}
				for (const auto& p : objPoints) {
					m_xs.push_back(p.x());
					m_ys.push_back(p.y());
					m_zs.push_back(p.z());
					m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);

				}
			}
			if (m_samplingEnabled) {
				size_t cptTrianglesCutOrNot = 0;
				for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++, cpt++) {
					if (cpt % nbForUpdate == 0) printf("\rDetermination of coloc locs for color %i: %.2f %%", n + 1, ((double)cpt / nbPs * 100.));
					for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j += 3) {
						if (!cutTriangles[cptTrianglesCutOrNot++]) continue;

						std::vector <poca::core::Vec3mf> queueTriangles = { dataO[j], dataO[j + 1], dataO[j + 2] };
						size_t first = 0, last = queueTriangles.size();
						for (uint32_t k = 0; k < m_subdiv3D; k++) {
							for (size_t cur = first; cur < last; cur += 3) {
								const poca::core::Vec3mf p1 = queueTriangles[cur], p2 = queueTriangles[cur + 1], p3 = queueTriangles[cur + 2], centroid = (p1 + p2 + p3) / 3.f;
	
								const double queryPt[3] = { centroid.x(), centroid.y(), centroid.z() };
								otherKdtree->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
								uint32_t idxDelaunayFace = otherD->indexTriangleOfPoint(centroid.x(), centroid.y(), centroid.z(), ret_index[0]);
								if (idxDelaunayFace != std::numeric_limits<std::uint32_t>::max() && selectionDelaunayElements[otherId][idxDelaunayFace] != std::numeric_limits<std::uint32_t>::max()) {
									m_xs.push_back(centroid.x());
									m_ys.push_back(centroid.y());
									m_zs.push_back(centroid.z());
									m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
								}

								//Add the 3 new triangles
								queueTriangles.push_back(p1);
								queueTriangles.push_back(p2);
								queueTriangles.push_back(centroid);
								queueTriangles.push_back(p2);
								queueTriangles.push_back(p3);
								queueTriangles.push_back(centroid);
								queueTriangles.push_back(p3);
								queueTriangles.push_back(p1);
								queueTriangles.push_back(centroid);
							}
							first = last;
							last = queueTriangles.size();
						}
					}
				}
			}
		}
		nbLocsObjs[n] = m_xs.size();
		firstIndexObjPerColor += m_nbObjectsPerColor[n];
		printf("\rDetermination of coloc locs for color %i: 100.00 %%\n", n + 1);
}

	poca::geometry::DelaunayTriangulationFactoryInterface* factory = poca::geometry::createDelaunayTriangulationFactory();
	m_delaunay = m_zs.empty() ? factory->createDelaunayTriangulation(m_xs, m_ys) : factory->createDelaunayTriangulation(m_xs, m_ys, m_zs);

	m_selectedTetrahedra.resize(m_delaunay->nbFaces());
	std::fill(m_selectedTetrahedra.begin(), m_selectedTetrahedra.end(), true);

	//Now, we need to determine which tetrahedra of the delaunay are part of coloc objects
	//To do so we compute the centroid of the tetrahedra and test if it belongs to objects of the two colors
	const std::vector<uint32_t>& triangles = m_delaunay->getTriangles();
	const poca::core::MyArrayUInt32& neighbors = m_delaunay->getNeighbors();
	const std::vector <uint32_t> indiceTriangles = neighbors.getFirstElements();
	const float* xs = m_delaunay->getXs();
	const float* ys = m_delaunay->getYs();
	const float* zs = m_delaunay->getZs();
	m_linkLocsToObjectsBothColor2.resize(m_xs.size());
	std::fill(m_linkLocsToObjectsBothColor2.begin(), m_linkLocsToObjectsBothColor2.end(), std::array<uint32_t, 2>{ std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max() });
	
	double nbPs = triangles.size() / 12;
	unsigned int nbForUpdate = nbPs / 100.;
	if (nbForUpdate == 0) nbForUpdate = 1;
	
	if (m_dimension == 2) {
		for (uint32_t n = 0, cpt = 0; n < triangles.size(); n += 3, cpt++) {
			size_t indexes[3] = { triangles[n], triangles[n + 1], triangles[n + 2] };
			poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
			for (size_t index : indexes)
				centroid += (poca::core::Vec3mf(xs[index], ys[index], 0.f) / 3.f);
			/*bool insideBoth = true;
			for (size_t i = 0; i < 2; i++) {
				const double queryPt[3] = { centroid.x(), centroid.y(), centroid.z() };
				kdtrees[i]->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
				uint32_t idxDelaunayFace = delaunays[i]->indexTriangleOfPoint(centroid.x(), centroid.y(), centroid.z(), ret_index[0]);
				insideBoth &= idxDelaunayFace != std::numeric_limits<std::uint32_t>::max() && selectionDelaunayElements[i][idxDelaunayFace] != std::numeric_limits<std::uint32_t>::max();
			}
			m_selectedTetrahedra[cpt] = insideBoth;*/
			uint32_t idObjectsInColors[2] = { std::numeric_limits<std::uint32_t>::max() , std::numeric_limits<std::uint32_t>::max() };
			for (size_t i = 0; i < 2; i++) {
				const double queryPt[3] = { centroid.x(), centroid.y(), centroid.z() };
				kdtrees[i]->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
				uint32_t idxDelaunayFace = delaunays[i]->indexTriangleOfPoint(centroid.x(), centroid.y(), centroid.z(), ret_index[0]);
				if (idxDelaunayFace == std::numeric_limits<std::uint32_t>::max()) continue;
				idObjectsInColors[i] = selectionDelaunayElements[i][idxDelaunayFace];
			}
			m_selectedTetrahedra[cpt] = idObjectsInColors[0] != std::numeric_limits<std::uint32_t>::max() && idObjectsInColors[1] != std::numeric_limits<std::uint32_t>::max();

			if (m_selectedTetrahedra[cpt]) {
				for (size_t index : indexes)
					m_linkLocsToObjectsBothColor2[index] = { idObjectsInColors[0], idObjectsInColors[1] };
			}
		}
	}
	else {
		for (uint32_t n = 0, cpt = 0; n < triangles.size(); n += 12, cpt++) {
			if (cpt % nbForUpdate == 0) printf("\rComputation of colocalization part 2: %.2f %%", ((double)cpt / nbPs * 100.));
			
			size_t indexes[4] = { triangles[n], triangles[n + 3 * 1], triangles[n + 3 * 2], triangles[n + 3 * 3] };
			poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
			for (size_t index : indexes)
				centroid += (poca::core::Vec3mf(xs[index], ys[index], zs[index]) / 4.f);
			uint32_t idObjectsInColors[2] = { std::numeric_limits<std::uint32_t>::max() , std::numeric_limits<std::uint32_t>::max() };
			for (size_t i = 0; i < 2; i++) {
				//const double queryPt[3] = { centroid.x(), centroid.y(), centroid.z() };
				//kdtrees[i]->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
				uint32_t idxDelaunayFace = delaunays[i]->indexTriangleOfPoint(centroid.x(), centroid.y(), centroid.z());
				if (idxDelaunayFace == std::numeric_limits<std::uint32_t>::max()) continue;
				idObjectsInColors[i] = selectionDelaunayElements[i][idxDelaunayFace];
			}
			m_selectedTetrahedra[cpt] = idObjectsInColors[0] != std::numeric_limits<std::uint32_t>::max() && idObjectsInColors[1] != std::numeric_limits<std::uint32_t>::max();

			if (m_selectedTetrahedra[cpt]) {
				for (size_t index : indexes)
					m_linkLocsToObjectsBothColor2[index] = { idObjectsInColors[0], idObjectsInColors[1] };
			}
		}
		printf("\rComputation of colocalization part 2: 100.00 %%\n");
	}

	poca::geometry::ObjectListFactory factoryObjects;
	//m_objectList = factoryObjects.createObjectListAlreadyIdentified(m_delaunay, m_selectionIntersection, FLT_MAX, m_minNbLocs);
	m_objectList = static_cast<poca::geometry::ObjectListDelaunay*>(factoryObjects.createObjectListFromDelaunay(m_delaunay, m_selectedTetrahedra, FLT_MAX, m_minNbLocs));
	//m_objectList = factoryObjects.createObjectList(m_delaunay, selectedLocsOverlapObjs, FLT_MAX, m_minNbLocs);
	m_objectList->setBoundingBox(m_delaunay->boundingBox());

	m_triangles = m_objectList->getTrianglesObjects().getData();
}

void ObjectColocalization::computeOverlap()
{
	poca::geometry::ObjectListDelaunay* objectsDelaunay[] = { static_cast<poca::geometry::ObjectListDelaunay*>(m_objectsPerColor[0]->currentObjectList()),  static_cast<poca::geometry::ObjectListDelaunay*>(m_objectsPerColor[1]->currentObjectList()) };
	if (objectsDelaunay[0] == NULL || objectsDelaunay[1] == NULL)
		return;

	poca::geometry::KdTree_DetectionPoint* kdtrees[2] = { NULL, NULL };
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("DetectionSet");
		poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
		if (dset) {
			kdtrees[n] = dset->getKdTree();
		}
	}
	poca::geometry::DelaunayTriangulationInterface* delaunays[2] = { NULL, NULL };
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("DelaunayTriangulation");
		poca::geometry::DelaunayTriangulationInterface* del = dynamic_cast <poca::geometry::DelaunayTriangulationInterface*>(bci);
		if (del)
			delaunays[n] = del;
	}
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("ObjectLists");
		poca::geometry::ObjectLists* obj = dynamic_cast <poca::geometry::ObjectLists*>(bci);
		if (obj)
			m_objectsPerColor[n] = obj;
	}
	m_dimension = m_objectsPerColor[0]->currentObjectList()->dimension();

	std::vector <uint32_t> selectionDelaunayElements[2];
	size_t nbLocsObjs[2], firstIndexObjPerColor = 0;

	for (size_t n = 0; n < 2; n++) {
		const float* xsobj = objectsDelaunay[n]->getXs(), * ysobj = objectsDelaunay[n]->getYs(), * zsobj = objectsDelaunay[n]->getZs();
		const poca::core::MyArrayUInt32& locs = m_objectsPerColor[n]->currentObjectList()->getLocsObjects();
		const std::vector <uint32_t>& data = locs.getData();
		const std::vector <uint32_t>& indexes = locs.getFirstElements();
		m_nbObjectsPerColor[n] = locs.nbElements();
		/*for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
			for (uint32_t j = indexes[i]; j < indexes[i + 1]; j++) {
				m_xs.push_back(xsobj[data[j]]);
				m_ys.push_back(ysobj[data[j]]);
				if (zsobj != NULL) m_zs.push_back(zsobj[data[j]]);
				m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
				//selectedLocs[data[j]] = i;
			}*/
		if (m_dimension == 2) {
			//Add some points on the objects contour, to properly sample them
			const poca::core::MyArrayVec3mf& outlines = m_objectsPerColor[n]->currentObjectList()->getOutlinesObjects();
			const std::vector <poca::core::Vec3mf>& dataO = outlines.getData();
			const std::vector <uint32_t>& indexesO = outlines.getFirstElements();
			for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
				for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j++) {
					const poca::core::Vec3mf& p = dataO[j];
					m_xs.push_back(p.x());
					m_ys.push_back(p.y());
					if (zsobj != NULL) m_zs.push_back(p.z());
					m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
				}
			if (m_samplingEnabled) {
				for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
					for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j++) {
						for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j += 2) {
							//for (size_t i = 0; i < dataO.size(); i += 2) {
							const poca::core::Vec3mf& p1 = dataO[j], & p2 = dataO[j + 1];
							poca::core::Vec3mf v = p2 - p1;
							float dvector = v.length(), d = m_distance2D;
							v.normalize();
							for (; d < dvector; d += m_distance2D) {
								poca::core::Vec3mf p = p1 + v * d;
								m_xs.push_back(p.x());
								m_ys.push_back(p.y());
								if (zsobj != NULL) m_zs.push_back(p.z());
								m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
							}
						}
					}
			}
		}
		else {
			//Add some points on the objects surface, to properly sample them
			const poca::core::MyArrayVec3mf& triangles = m_objectsPerColor[n]->currentObjectList()->getTrianglesObjects();
			const std::vector <poca::core::Vec3mf>& dataO = triangles.getData();
			const std::vector <uint32_t>& indexesO = triangles.getFirstElements();
			//Only contours of the objects
			/*for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++) {
				std::set <poca::core::Vec3mf> objPoints;
				for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j++) {
					objPoints.insert(dataO[j]);
				}
				for (const auto& p : objPoints) {
					m_xs.push_back(p.x());
					m_ys.push_back(p.y());
					m_zs.push_back(p.z());
					m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);

				}
			}*/
			//All objects
			for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
				for (uint32_t j = indexes[i]; j < indexes[i + 1]; j++) {
					m_xs.push_back(xsobj[data[j]]);
					m_ys.push_back(ysobj[data[j]]);
					if (zsobj != NULL) m_zs.push_back(zsobj[data[j]]);
					m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
					//selectedLocs[data[j]] = i;
				}
			if (m_samplingEnabled) {
				for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
					for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j += 3) {
						std::vector <poca::core::Vec3mf> queueTriangles = { dataO[j], dataO[j + 1], dataO[j + 2] };
						size_t first = 0, last = queueTriangles.size();
						for (uint32_t k = 0; k < m_subdiv3D; k++) {
							for (size_t cur = first; cur < last; cur += 3) {
								const poca::core::Vec3mf p1 = queueTriangles[cur], p2 = queueTriangles[cur + 1], p3 = queueTriangles[cur + 2], centroid = (p1 + p2 + p3) / 3.f;
								m_xs.push_back(centroid.x());
								m_ys.push_back(centroid.y());
								m_zs.push_back(centroid.z());
								m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);

								//Add the 3 new triangles
								queueTriangles.push_back(p1);
								queueTriangles.push_back(p2);
								queueTriangles.push_back(centroid);
								queueTriangles.push_back(p2);
								queueTriangles.push_back(p3);
								queueTriangles.push_back(centroid);
								queueTriangles.push_back(p3);
								queueTriangles.push_back(p1);
								queueTriangles.push_back(centroid);
							}
							first = last;
							last = queueTriangles.size();
						}
					}
			}
		}
		nbLocsObjs[n] = m_xs.size();
		firstIndexObjPerColor += m_nbObjectsPerColor[n];
		selectionDelaunayElements[n] = objectsDelaunay[n]->getLinkTriangulationFacesToObjects();
	}

	/*uint32_t idD = 1;
	poca::geometry::DelaunayTriangulationInterface* delau = delaunays[idD];
	const std::vector<uint32_t>& triangles = delau->getTriangles();
	const float* xsd = delau->getXs(), * ysd = delau->getYs(), * zsd = delau->getZs();
	for (size_t n = 0, cpt = 0; n < triangles.size(); n+=3, cpt++) {
		if (selectionDelaunayElements[idD][cpt] == std::numeric_limits<std::uint32_t>::max()) continue;
		delau->getVerticesTriangle(cpt, m_triangles);
	}*/

	/*std::ofstream fs("C:/DevC++/poca/bin/poca/overlap.csv");
	fs << "x,y,z" << std::endl;
	for (size_t n = 0; n < m_xs.size(); n++)
		fs << m_xs[n] << "," << m_ys[n] << "," << m_zs[n] << std::endl;
	fs.close();*/

	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	poca::geometry::DelaunayTriangulationFactoryInterface* factory = poca::geometry::createDelaunayTriangulationFactory();
	m_delaunay = m_zs.empty() ? factory->createDelaunayTriangulation(m_xs, m_ys) : factory->createDelaunayTriangulation(m_xs, m_ys, m_zs);
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;

	m_selectionIntersection.resize(m_xs.size());
	std::fill(m_selectionIntersection.begin(), m_selectionIntersection.end(), std::numeric_limits<std::uint32_t>::max());
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	double nbPs = m_xs.size();
	unsigned int nbForUpdate = nbPs / 100.;
	if (nbForUpdate == 0) nbForUpdate = 1;
	const std::size_t num_results = 1;
	std::vector<size_t> ret_index(num_results);
	std::vector<double> out_dist_sqr(num_results);
	const uint32_t SELECTED_VALUE = std::numeric_limits<std::uint32_t>::max() - 1;
	printf("\rComputing intersection selection part 1: %.2f %%", (0. / nbPs * 100.));
	for (size_t n = 0; n < m_xs.size(); n++) {
		if (n % nbForUpdate == 0) printf("\rComputing intersection selection part 1: %.2f %%", ((double)n / nbPs * 100.));
		uint32_t idxLocInColor = (n < nbLocsObjs[0]) ? n : n - nbLocsObjs[0], otherId = (n < nbLocsObjs[0]) ? 1 : 0;
		//std::cout << idxLocInColor << " - ";
		poca::geometry::DelaunayTriangulationInterface* otherD = delaunays[otherId];
		poca::geometry::KdTree_DetectionPoint* otherKdtree = kdtrees[otherId];
		float x = m_xs[n], y = m_ys[n], z = m_zs.empty() ? 0.f : m_zs[n];
		const double queryPt[3] = { x, y, z };
		otherKdtree->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
		uint32_t idxDelaunayFace = otherD->indexTriangleOfPoint(x, y, z, ret_index[0]);
		if (idxDelaunayFace != std::numeric_limits<std::uint32_t>::max() && selectionDelaunayElements[otherId][idxDelaunayFace] != std::numeric_limits<std::uint32_t>::max())
			m_selectionIntersection[n] = SELECTED_VALUE;
	}
	std::vector <uint32_t> tmpSelectPoints = m_selectionIntersection;
	printf("\rComputing intersection selection part 1: 100.00 %%\n");

	//Now tag every selected intersection locs with an id corresponding to an object id
	//We need to go through all locs, test if the loc is part on an intersection, and go through its neighs
	//to tag them with the same obj id
	const std::vector<uint32_t>& triangles = m_delaunay->getTriangles(); 
	const poca::core::MyArrayUInt32& neighbors = m_delaunay->getNeighbors();
	const std::vector <uint32_t> indiceTriangles = neighbors.getFirstElements();
	const float* xs = m_delaunay->getXs();
	const float* ys = m_delaunay->getYs();
	const float* zs = m_delaunay->getZs();
	std::vector <bool> selectedTriangles(m_delaunay->nbFaces(), false);
	std::vector <bool> tmpSelection;
	uint32_t cptObj = 0;
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;

	std::vector <bool> selectedLocsOverlapObjs(m_xs.size(), false);
	for (size_t n = 0; n < selectedLocsOverlapObjs.size(); n++)
		selectedLocsOverlapObjs[n] = m_selectionIntersection[n] == SELECTED_VALUE;

	if (m_dimension == 2) {
		//First, identify all triangles that have all their locs SELECTED
		for (uint32_t n = 0, cpt = 0; n < triangles.size(); n += 3, cpt++) {
			size_t i1 = triangles[n], i2 = triangles[n + 1], i3 = triangles[n + 2];
			uint32_t o1 = m_selectionIntersection[i1], o2 = m_selectionIntersection[i2], o3 = m_selectionIntersection[i3];
			selectedTriangles[cpt] = o1 == o2 && o2 == o3 && o1 == SELECTED_VALUE;
		}
		std::vector <uint32_t> neighsTriangle;
		for (uint32_t n = 0; n < m_delaunay->nbFaces(); n++) {
			if (!selectedTriangles[n]) continue;
			std::vector <uint32_t> queueTri = { n };
			selectedTriangles[n] = false;
			size_t cur = 0, size = queueTri.size();
			while (cur < size) {
				uint32_t curTri = queueTri[cur];
				for (uint32_t i = 0; i < neighbors.nbElementsObject(curTri); i++) {
					uint32_t indexNeigh = neighbors.elementIObject(curTri, i);
					if (selectedTriangles[indexNeigh]) {
						queueTri.push_back(indexNeigh);
						selectedTriangles[indexNeigh] = false;
					}
				}
				size = queueTri.size();
				cur++;
			}
			uint32_t cptt = 0;
			for (uint32_t idTri : queueTri) {
				uint32_t idLocs[] = { triangles[idTri * 3] , triangles[idTri * 3 + 1] , triangles[idTri * 3 + 2] };
				for (uint32_t id : idLocs)
					if (m_selectionIntersection[id] == SELECTED_VALUE){
						m_selectionIntersection[id] = cptObj;
						cptt++;
					}
			}
			cptObj++;
		}
	}
	else {
		//First, identify all triangles that have all their locs SELECTED
		for (size_t n = 0, cpt = 0; n < triangles.size(); n += 12, cpt++) {
			size_t i1 = triangles[n], i2 = triangles[n + 3 * 1], i3 = triangles[n + 3 * 2], i4 = triangles[n + 3 * 3];
			uint32_t o1 = m_selectionIntersection[i1], o2 = m_selectionIntersection[i2], o3 = m_selectionIntersection[i3], o4 = m_selectionIntersection[i4];
			selectedTriangles[cpt] = (o1 == o2 && o2 == o3 && o3 == o4 && o1 == SELECTED_VALUE);
		}
		m_selectedTetrahedra = selectedTriangles;
		tmpSelection = selectedTriangles;
		std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
		nbPs = m_delaunay->nbFaces();
		nbForUpdate = nbPs / 100.;
		if (nbForUpdate == 0) nbForUpdate = 1;
		printf("\rComputing intersection selection part 2: %.2f %%", (0. / nbPs * 100.));
		std::vector <uint32_t> neighsTriangle;
		for (uint32_t n = 0; n < m_delaunay->nbFaces(); n++) {
			if (n % nbForUpdate == 0) printf("\rComputing intersection selection part 2: %.2f %%", ((double)n / nbPs * 100.));
			if (!selectedTriangles[n]) continue;
			std::vector <uint32_t> queueTri = { n };
			selectedTriangles[n] = false;
			size_t cur = 0, size = queueTri.size();
			while (cur < size) {
				uint32_t curTri = queueTri[cur];
				for (uint32_t i = 0; i < neighbors.nbElementsObject(curTri); i++) {
					uint32_t indexNeigh = neighbors.elementIObject(curTri, i);
					if (selectedTriangles[indexNeigh]) {
						queueTri.push_back(indexNeigh);
						selectedTriangles[indexNeigh] = false;
					}
				}
				size = queueTri.size();
				cur++;
			}
			//Determine if the overlap object is just composed of one color -> discard if it is the case
			bool hasColor1 = false, hasColor2 = false;
			for (uint32_t idTri : queueTri) {
				uint32_t index = indiceTriangles[idTri];
				uint32_t idLocs[] = { triangles[3 * index] , triangles[3 * index + 3 * 1] , triangles[3 * index + 3 * 2] , triangles[3 * index + 3 * 3] };
				for (uint32_t id : idLocs) {
					if (id < nbLocsObjs[0])
						hasColor1 = true;
					else
						hasColor2 = true;
				}
			}
			if (hasColor1 && hasColor2) {
				//Create objects
				uint32_t cptt = 0;
				for (uint32_t idTri : queueTri) {
					uint32_t index = indiceTriangles[idTri];
					uint32_t idLocs[] = { triangles[3 * index] , triangles[3 * index + 3 * 1] , triangles[3 * index + 3 * 2] , triangles[3 * index + 3 * 3] };
					for (uint32_t id : idLocs) {
						if (m_selectionIntersection[id] == SELECTED_VALUE) 
						{
							m_selectionIntersection[id] = cptObj;
							cptt++;
						}
						//else
						//	std::cout << "Really?" << std::endl;
					}
				}
			}
			else {
				//Discard the corresponding locs since the overlap is only composed of one color
				for (uint32_t idTri : queueTri) {
					uint32_t index = indiceTriangles[idTri];
					uint32_t idLocs[] = { triangles[3 * index] , triangles[3 * index + 3 * 1] , triangles[3 * index + 3 * 2] , triangles[3 * index + 3 * 3] };
					for (uint32_t id : idLocs)
						if (m_selectionIntersection[id] == SELECTED_VALUE)
							m_selectionIntersection[id] = std::numeric_limits<std::uint32_t>::max();
					tmpSelection[idTri] = false;
				}
			}

			//std::cout << "cell " << n << ", # queue = " << queueTri.size() << ", obj idx = " << cptObj << std::endl;
			cptObj++;
		}
		printf("\rComputing intersection selection part 1: 100.00 %%\n");
	}
	/*for (uint32_t n = 0; n < m_xs.size(); n++) {
		if (m_selectionIntersection[n] != SELECTED_VALUE) continue;
		std::vector <uint32_t> queueTri = { n };
		size_t cur = 0, size = queueTri.size();
		while (cur < size) {
			uint32_t curTri = queueTri[cur];
			m_selectionIntersection[curTri] = cptObj;
			for (uint32_t i = 0; i < neighbors.nbElementsObject(curTri); i++) {
				uint32_t indexNeigh = neighbors.elementIObject(curTri, i);
				if (m_selectionIntersection[indexNeigh] == SELECTED_VALUE)
					queueTri.push_back(indexNeigh);
			}
			size = queueTri.size();
			cur++;
		}
		cptObj++;
	}*/

	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	for (uint32_t n = 0; n < m_selectionIntersection.size(); n++)
		if (m_selectionIntersection[n] == SELECTED_VALUE)
			m_selectionIntersection[n] = std::numeric_limits<std::uint32_t>::max();

	//Now we can use m_selectionIntersection to create an objectlist of the overlaps
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	poca::geometry::ObjectListFactory factoryObjects;
	//m_objectList = factoryObjects.createObjectListAlreadyIdentified(m_delaunay, m_selectionIntersection, FLT_MAX, m_minNbLocs);
	m_objectList = static_cast<poca::geometry::ObjectListDelaunay*>(factoryObjects.createObjectListFromDelaunay(m_delaunay, tmpSelection, FLT_MAX, m_minNbLocs));
	//m_objectList = factoryObjects.createObjectList(m_delaunay, selectedLocsOverlapObjs, FLT_MAX, m_minNbLocs);
	m_objectList->setBoundingBox(m_delaunay->boundingBox());

	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	m_triangles = m_objectList->getTrianglesObjects().getData();
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;

	m_selectionIntersection = tmpSelectPoints;
}

void ObjectColocalization::computeOverlap2()
{
	poca::geometry::KdTree_DetectionPoint* kdtrees[2] = { NULL, NULL };
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("DetectionSet");
		poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
		if (dset) {
			kdtrees[n] = dset->getKdTree();
		}
	}
	poca::geometry::DelaunayTriangulationInterface* delaunays[2] = { NULL, NULL };
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("DelaunayTriangulation");
		poca::geometry::DelaunayTriangulationInterface* del = dynamic_cast <poca::geometry::DelaunayTriangulationInterface*>(bci);
		if (del)
			delaunays[n] = del;
	}
	poca::geometry::ObjectListDelaunay* objLists[2] = { NULL, NULL };
	for (size_t n = 0; n < 2; n++) {
		poca::core::BasicComponentInterface* bci = m_objects[n]->getBasicComponent("ObjectListDelaunay");
		poca::geometry::ObjectListDelaunay* obj = dynamic_cast <poca::geometry::ObjectListDelaunay*>(bci);
		if (obj)
			objLists[n] = obj;
	}
	m_dimension = objLists[0]->dimension();

	std::vector <uint32_t> selectionDelaunayElements[2];
	size_t nbLocsObjs[2], firstIndexObjPerColor = 0;

	for (size_t n = 0; n < 2; n++) {
		const float* xsobj = objLists[n]->getXs(), * ysobj = objLists[n]->getYs(), * zsobj = objLists[n]->getZs();
		const poca::core::MyArrayUInt32& locs = objLists[n]->getLocsObjects();
		const std::vector <uint32_t>& data = locs.getData();
		const std::vector <uint32_t>& indexes = locs.getFirstElements();
		m_nbObjectsPerColor[n] = locs.nbElements();
		/*for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
			for (uint32_t j = indexes[i]; j < indexes[i + 1]; j++) {
				m_xs.push_back(xsobj[data[j]]);
				m_ys.push_back(ysobj[data[j]]);
				if (zsobj != NULL) m_zs.push_back(zsobj[data[j]]);
				m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
				//selectedLocs[data[j]] = i;
			}*/
		if (m_dimension == 2) {
			//Add some points on the objects contour, to properly sample them
			const poca::core::MyArrayVec3mf& outlines = objLists[n]->getOutlinesObjects();
			const std::vector <poca::core::Vec3mf>& dataO = outlines.getData();
			const std::vector <uint32_t>& indexesO = outlines.getFirstElements();
			float nbSamples = 10, step = 0.5f;
			for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
				for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j += 2) {
					//for (size_t i = 0; i < dataO.size(); i += 2) {
					const poca::core::Vec3mf& p1 = dataO[j], & p2 = dataO[j + 1];
					poca::core::Vec3mf v = p2 - p1;
					float dvector = v.length(), d = step;
					v.normalize();
					for (; d < dvector; d += step) {
						poca::core::Vec3mf p = p1 + v * d;
						m_xs.push_back(p.x());
						m_ys.push_back(p.y());
						if (zsobj != NULL) m_zs.push_back(p.z());
						m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);
					}
				}
		}
		else {
			/*//Add some points on the objects surface, to properly sample them
			const poca::core::MyArrayVec3mf& triangles = objLists[n]->getTrianglesObjects();
			const std::vector <poca::core::Vec3mf>& dataO = triangles.getData();
			const std::vector <uint32_t>& indexesO = triangles.getFirstElements();
			//float nbSamples = 10, step = 0.5f;
			uint32_t nbSub = 2;
			for (size_t i = 0; i < m_nbObjectsPerColor[n]; i++)
				for (uint32_t j = indexesO[i]; j < indexesO[i + 1]; j += 3) {
					std::vector <poca::core::Vec3mf> queueTriangles = { dataO[j], dataO[j + 1], dataO[j + 2] };
					size_t first = 0, last = queueTriangles.size();
					for (uint32_t k = 0; k < nbSub; k++) {
						for (size_t cur = first; cur < last; cur += 3) {
							const poca::core::Vec3mf p1 = queueTriangles[cur], p2 = queueTriangles[cur + 1], p3 = queueTriangles[cur + 2], centroid = (p1 + p2 + p3) / 3.f;
							m_xs.push_back(centroid.x());
							m_ys.push_back(centroid.y());
							m_zs.push_back(centroid.z());
							m_linkLocsToObjectsBothColor.push_back(firstIndexObjPerColor + i);

							//Add the 3 new triangles
							queueTriangles.push_back(p1);
							queueTriangles.push_back(p2);
							queueTriangles.push_back(centroid);
							queueTriangles.push_back(p2);
							queueTriangles.push_back(p3);
							queueTriangles.push_back(centroid);
							queueTriangles.push_back(p3);
							queueTriangles.push_back(p1);
							queueTriangles.push_back(centroid);
						}
						first = last;
						last = queueTriangles.size();
					}
				}*/
		}
		nbLocsObjs[n] = m_xs.size();
		firstIndexObjPerColor += m_nbObjectsPerColor[n];
		selectionDelaunayElements[n] = objLists[n]->getLinkTriangulationFacesToObjects();
	}

	/*uint32_t idD = 1;
	poca::geometry::DelaunayTriangulationInterface* delau = delaunays[idD];
	const std::vector<uint32_t>& triangles = delau->getTriangles();
	const float* xsd = delau->getXs(), * ysd = delau->getYs(), * zsd = delau->getZs();
	for (size_t n = 0, cpt = 0; n < triangles.size(); n+=3, cpt++) {
		if (selectionDelaunayElements[idD][cpt] == std::numeric_limits<std::uint32_t>::max()) continue;
		delau->getVerticesTriangle(cpt, m_triangles);
	}*/

	/*std::ofstream fs("C:/DevC++/poca/bin/poca/overlap.csv");
	fs << "x,y,z" << std::endl;
	for (size_t n = 0; n < m_xs.size(); n++)
		fs << m_xs[n] << "," << m_ys[n] << "," << m_zs[n] << std::endl;
	fs.close();*/

	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	poca::geometry::DelaunayTriangulationFactoryInterface* factory = poca::geometry::createDelaunayTriangulationFactory();
	m_delaunay = m_zs.empty() ? factory->createDelaunayTriangulation(m_xs, m_ys) : factory->createDelaunayTriangulation(m_xs, m_ys, m_zs);
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;

	m_selectionIntersection.resize(m_xs.size());
	std::fill(m_selectionIntersection.begin(), m_selectionIntersection.end(), std::numeric_limits<std::uint32_t>::max());
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	double nbPs = m_xs.size();
	unsigned int nbForUpdate = nbPs / 100.;
	if (nbForUpdate == 0) nbForUpdate = 1;
	const std::size_t num_results = 1;
	std::vector<size_t> ret_index(num_results);
	std::vector<double> out_dist_sqr(num_results);
	const uint32_t SELECTED_VALUE = std::numeric_limits<std::uint32_t>::max() - 1;
	printf("\rComputing intersection selection part 1: %.2f %%", (0. / nbPs * 100.));
	for (size_t n = 0; n < m_xs.size(); n++) {
		if (n % nbForUpdate == 0) printf("\rComputing intersection selection part 1: %.2f %%", ((double)n / nbPs * 100.));
		uint32_t idxLocInColor = (n < nbLocsObjs[0]) ? n : n - nbLocsObjs[0], otherId = (n < nbLocsObjs[0]) ? 1 : 0;
		//std::cout << idxLocInColor << " - ";
		poca::geometry::DelaunayTriangulationInterface* otherD = delaunays[otherId];
		poca::geometry::KdTree_DetectionPoint* otherKdtree = kdtrees[otherId];
		float x = m_xs[n], y = m_ys[n], z = m_zs.empty() ? 0.f : m_zs[n];
		const double queryPt[3] = { x, y, z };
		otherKdtree->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
		uint32_t idxDelaunayFace = otherD->indexTriangleOfPoint(x, y, z, ret_index[0]);
		if (idxDelaunayFace != std::numeric_limits<std::uint32_t>::max() && selectionDelaunayElements[otherId][idxDelaunayFace] != std::numeric_limits<std::uint32_t>::max())
			m_selectionIntersection[n] = SELECTED_VALUE;
	}
	printf("\rComputing intersection selection part 1: 100.00 %%\n");

	//Now tag every selected intersection locs with an id corresponding to an object id
	//We need to go through all locs, test if the loc is part on an intersection, and go through its neighs
	//to tag them with the same obj id
	const std::vector<uint32_t>& triangles = m_delaunay->getTriangles();
	const poca::core::MyArrayUInt32& neighbors = m_delaunay->getNeighbors();
	const std::vector <uint32_t> indiceTriangles = neighbors.getFirstElements();
	const float* xs = m_delaunay->getXs();
	const float* ys = m_delaunay->getYs();
	const float* zs = m_delaunay->getZs();
	std::vector <bool> selectedTriangles(m_delaunay->nbFaces(), false);
	uint32_t cptObj = 0;
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	if (m_dimension == 2) {
		//First, identify all triangles that have all their locs SELECTED
		for (uint32_t n = 0, cpt = 0; n < triangles.size(); n += 3, cpt++) {
			size_t i1 = triangles[n], i2 = triangles[n + 1], i3 = triangles[n + 2];
			uint32_t o1 = m_selectionIntersection[i1], o2 = m_selectionIntersection[i2], o3 = m_selectionIntersection[i3];
			selectedTriangles[cpt] = o1 == o2 && o2 == o3 && o1 == SELECTED_VALUE;
		}
		std::vector <uint32_t> neighsTriangle;
		for (uint32_t n = 0; n < m_delaunay->nbFaces(); n++) {
			if (!selectedTriangles[n]) continue;
			std::vector <uint32_t> queueTri = { n };
			size_t cur = 0, size = queueTri.size();
			while (cur < size) {
				uint32_t curTri = queueTri[cur];
				selectedTriangles[curTri] = false;
				for (uint32_t i = 0; i < neighbors.nbElementsObject(curTri); i++) {
					uint32_t indexNeigh = neighbors.elementIObject(curTri, i);
					if (selectedTriangles[indexNeigh])
						queueTri.push_back(indexNeigh);
				}
				size = queueTri.size();
				cur++;
			}
			uint32_t cptt = 0;
			for (uint32_t idTri : queueTri) {
				uint32_t idLocs[] = { triangles[idTri * 3] , triangles[idTri * 3 + 1] , triangles[idTri * 3 + 2] };
				for (uint32_t id : idLocs)
					if (m_selectionIntersection[id] == SELECTED_VALUE) {
						m_selectionIntersection[id] = cptObj;
						cptt++;
					}
			}
			cptObj++;
		}
	}
	else {
		//First, identify all triangles that have all their locs SELECTED
		for (size_t n = 0, cpt = 0; n < triangles.size(); n += 12, cpt++) {
			size_t i1 = triangles[n], i2 = triangles[n + 3 * 1], i3 = triangles[n + 3 * 2], i4 = triangles[n + 3 * 3];
			uint32_t o1 = m_selectionIntersection[i1], o2 = m_selectionIntersection[i2], o3 = m_selectionIntersection[i3], o4 = m_selectionIntersection[i4];
			selectedTriangles[cpt] = (o1 == o2 && o2 == o3 && o3 == o4 && o1 == SELECTED_VALUE);
		}
		std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
		nbPs = m_delaunay->nbFaces();
		nbForUpdate = nbPs / 100.;
		if (nbForUpdate == 0) nbForUpdate = 1;
		printf("\rComputing intersection selection part 2: %.2f %%", (0. / nbPs * 100.));
		std::vector <uint32_t> neighsTriangle;
		for (uint32_t n = 0; n < m_delaunay->nbFaces(); n++) {
			if (n % nbForUpdate == 0) printf("\rComputing intersection selection part 2: %.2f %%", ((double)n / nbPs * 100.));
			if (!selectedTriangles[n]) continue;
			std::vector <uint32_t> queueTri = { n };
			size_t cur = 0, size = queueTri.size();
			while (cur < size) {
				uint32_t curTri = queueTri[cur];
				selectedTriangles[curTri] = false;
				for (uint32_t i = 0; i < neighbors.nbElementsObject(curTri); i++) {
					uint32_t indexNeigh = neighbors.elementIObject(curTri, i);
					if (selectedTriangles[indexNeigh])
						queueTri.push_back(indexNeigh);
				}
				size = queueTri.size();
				cur++;
			}
			uint32_t cptt = 0;
			for (uint32_t idTri : queueTri) {
				uint32_t index = indiceTriangles[idTri];
				uint32_t idLocs[] = { triangles[3 * index] , triangles[3 * index + 3 * 1] , triangles[3 * index + 3 * 2] , triangles[3 * index + 3 * 3] };
				for (uint32_t id : idLocs)
					if (m_selectionIntersection[id] == SELECTED_VALUE) {
						m_selectionIntersection[id] = cptObj;
						cptt++;
					}
			}
			std::cout << "cell " << n << ", # queue = " << queueTri.size() << ", obj idx = " << cptObj << std::endl;
			cptObj++;
		}
		printf("\rComputing intersection selection part 1: 100.00 %%\n");
	}
	/*for (uint32_t n = 0; n < m_xs.size(); n++) {
		if (m_selectionIntersection[n] != SELECTED_VALUE) continue;
		std::vector <uint32_t> queueTri = { n };
		size_t cur = 0, size = queueTri.size();
		while (cur < size) {
			uint32_t curTri = queueTri[cur];
			m_selectionIntersection[curTri] = cptObj;
			for (uint32_t i = 0; i < neighbors.nbElementsObject(curTri); i++) {
				uint32_t indexNeigh = neighbors.elementIObject(curTri, i);
				if (m_selectionIntersection[indexNeigh] == SELECTED_VALUE)
					queueTri.push_back(indexNeigh);
			}
			size = queueTri.size();
			cur++;
		}
		cptObj++;
	}*/

	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	for (uint32_t n = 0; n < m_selectionIntersection.size(); n++)
		if (m_selectionIntersection[n] == SELECTED_VALUE)
			m_selectionIntersection[n] = std::numeric_limits<std::uint32_t>::max();

	//Now we can use m_selectionIntersection to create an objectlist of the overlaps
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	poca::geometry::ObjectListFactory factoryObjects;
	m_objectList = static_cast<poca::geometry::ObjectListDelaunay*>(factoryObjects.createObjectListAlreadyIdentified(m_delaunay, m_selectionIntersection, FLT_MAX, 5));
	m_objectList->setBoundingBox(m_delaunay->boundingBox());

	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
	m_triangles = m_objectList->getTrianglesObjects().getData();
	std::cout << __FUNCTION__ << " - " << __LINE__ << std::endl;
}

poca::core::BoundingBox ObjectColocalization::computeBoundingBoxElement(const int _idx) const
{
	poca::core::BoundingBox bbox;
	if (m_zs.empty())
		bbox.set(FLT_MAX, FLT_MAX, 0.f, -FLT_MAX, -FLT_MAX, 0.f);
	else
		bbox.set(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

	const poca::core::MyArrayVec3mf& triangles = m_objectList->getTrianglesObjects();
	for (size_t j = 0; j < triangles.nbElementsObject(_idx); j++) {
		const poca::core::Vec3mf& vertex = triangles.elementIObject(_idx, j);
		float x = vertex.x(), y = vertex.y();
		bbox[0] = x < bbox[0] ? x : bbox[0];
		bbox[1] = y < bbox[1] ? y : bbox[1];

		bbox[3] = x > bbox[3] ? x : bbox[3];
		bbox[4] = y > bbox[4] ? y : bbox[4];

		if (!m_zs.empty()) {
			float z = m_zs.empty() ? 0.f : vertex.z();
			bbox[2] = z < bbox[2] ? z : bbox[2];
			bbox[5] = z > bbox[5] ? z : bbox[5];
		}
	}
	/*const poca::core::MyArrayUInt32& locs = m_objectList->getLocsObjects();
	for (size_t j = 0; j < locs.nbElementsObject(_idx); j++) {
		uint32_t vertex = locs.elementIObject(_idx, j);
		float x = m_xs[vertex], y = m_ys[vertex];
		bbox[0] = x < bbox[0] ? x : bbox[0];
		bbox[1] = y < bbox[1] ? y : bbox[1];

		bbox[3] = x > bbox[3] ? x : bbox[3];
		bbox[4] = y > bbox[4] ? y : bbox[4];

		if (!m_zs.empty()) {
			float z = m_zs.empty() ? 0.f : m_zs[vertex];
			bbox[2] = z < bbox[2] ? z : bbox[2];
			bbox[5] = z > bbox[5] ? z : bbox[5];
		}
	}*/

	return bbox;
}

const float ObjectColocalization::getAreaObjectColor(const uint32_t _idxColor, const uint32_t _idxObj) const
{
	assert(_idxColor < 2 && _idxObj < m_objectsPerColor[_idxColor]->currentObjectList()->nbObjects());
	const std::vector <float>& areas = m_objectsPerColor[_idxColor]->currentObjectList()->dimension() == 2 ? m_objectsPerColor[_idxColor]->currentObjectList()->getMyData("area")->getData<float>() : m_objectsPerColor[_idxColor]->currentObjectList()->getMyData("volume")->getData<float>();
	return areas[_idxObj];
}

const float ObjectColocalization::getAreaObjectIntersection(const uint32_t _idxObj) const
{
	assert(_idxObj < m_objectList->nbObjects());
	const std::vector <float>& areas = m_objectList->dimension() == 2 ? m_objectList->getMyData("area")->getData<float>() : m_objectList->getMyData("volume")->getData<float>();
	return areas[_idxObj];
}