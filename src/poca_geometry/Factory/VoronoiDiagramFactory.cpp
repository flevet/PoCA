/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramFactory.cpp
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

#include <Windows.h>
#include <GL/glew.h>
#include <omp.h>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#ifndef NO_CUDA
#include <cuda_runtime.h>
#endif

#include <QtWidgets/QMessageBox>

#include <General/Vec3.hpp>
#include <General/MyArray.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <General/Histogram.hpp>
#include <General/PluginList.hpp>

#include "VoronoiDiagramFactory.hpp"
#include "DelaunayTriangulationFactory.hpp"
#include "../Geometry/DelaunayTriangulation.hpp"
#include "../Geometry/VoronoiDiagram.hpp"
#include "../Geometry/voronator.hpp"
#include "../Geometry/CGAL_includes.hpp"
#include "../3D_voronoi_GPU/voronoi.h"

#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

namespace poca::geometry {
	VoronoiDiagramFactoryInterface* createVoronoiDiagramFactory() {
		return new VoronoiDiagramFactory();
	}

	VoronoiDiagramFactory::VoronoiDiagramFactory()
	{

	}

	VoronoiDiagramFactory::~VoronoiDiagramFactory()
	{

	}

	VoronoiDiagram* VoronoiDiagramFactory::createVoronoiDiagram(poca::core::MyObjectInterface* _obj, bool _noCells, poca::core::PluginList* _plugins, const bool _addCommands)
	{
		if (_obj == NULL) return NULL;

		poca::geometry::VoronoiDiagram* voro = nullptr;
		poca::core::BasicComponentInterface* bci = _obj->getBasicComponent("DetectionSet");
		if (bci == NULL)
			return NULL;
		poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
		if (dset == NULL)
			return NULL;
		bci = _obj->getBasicComponent("DelaunayTriangulation");
		poca::geometry::DelaunayTriangulationInterface* delaunay = dynamic_cast <poca::geometry::DelaunayTriangulationInterface*>(bci);
		if (delaunay == NULL || delaunay && dynamic_cast <DelaunayTriangulation2DOnSphere*>(delaunay)) {
			dset->executeCommand(false, "computeDelaunay");
			bci = _obj->getBasicComponent("DelaunayTriangulation");
			delaunay = dynamic_cast <poca::geometry::DelaunayTriangulationInterface*>(bci);
			if (!delaunay)
				return NULL;
		}
		//std::ofstream fs("e:/timings.txt", std::fstream::out | std::fstream::app);
		clock_t t1, t2;
		t1 = clock();
		poca::geometry::KdTree_DetectionPoint* kdtree = dset->getKdTree();
		const std::vector <float>& xs = static_cast<poca::core::Histogram<float>*>(dset->getOriginalHistogram("x"))->getValues();
		const std::vector <float>& ys = static_cast<poca::core::Histogram<float>*>(dset->getOriginalHistogram("y"))->getValues();
		if (!dset->hasData("z")) {
			voro = createVoronoiDiagram(xs, ys, dset->boundingBox(), kdtree, delaunay);
		}
		else {
			const std::vector <float>& zs = static_cast<poca::core::Histogram<float>*>(dset->getOriginalHistogram("z"))->getValues();
			voro = createVoronoiDiagram(xs, ys, zs, kdtree, delaunay, _noCells);
		}
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		//fs << elapsed << "\t";
		//fs.close();
		if (voro != NULL) {
			voro->setBoundingBox(dset->boundingBox());
			if(_addCommands)
				_plugins->addCommands(voro);
			_obj->addBasicComponent(voro);
			if (voro != NULL) {
				_obj->notify("LoadObjCharacteristicsAllWidgets");
				_obj->notifyAll("updateDisplay");
			}
		}
		t2 = clock();
		elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("time for computing Voronoi: %ld ms\n", elapsed);
		return voro;
	}

	VoronoiDiagram* VoronoiDiagramFactory::createVoronoiDiagramOnSphere(poca::core::MyObjectInterface* _obj, bool _noCells, poca::core::PluginList* _plugins, const bool _addCommands)
	{
		if (_obj == NULL) return NULL;

		poca::geometry::VoronoiDiagram* voro = nullptr;
		poca::core::BasicComponentInterface* bci = _obj->getBasicComponent("DetectionSet");
		if (bci == NULL)
			return NULL;
		poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
		if (dset == NULL)
			return NULL;
		bci = _obj->getBasicComponent("DelaunayTriangulation");
		poca::geometry::DelaunayTriangulationInterface* delaunay = dynamic_cast <poca::geometry::DelaunayTriangulationInterface*>(bci);
		if (delaunay == NULL || delaunay && !dynamic_cast <DelaunayTriangulation2DOnSphere*>(delaunay)) {
			dset->executeCommand(false, "computeDelaunay", "onSphere", true);
			bci = _obj->getBasicComponent("DelaunayTriangulation");
			if (bci == NULL)
				return NULL;
			delaunay = dynamic_cast <poca::geometry::DelaunayTriangulationInterface*>(bci);
			if (!delaunay)
				return NULL;
		}
		clock_t t1, t2;
		t1 = clock();
		poca::geometry::KdTree_DetectionPoint* kdtree = dset->getKdTree();
		const std::vector <float>& xs = delaunay->xs();
		const std::vector <float>& ys = delaunay->ys();
		const std::vector <float>& zs = delaunay->zs();
		if (zs.empty()) return NULL;
		voro = createVoronoiDiagramOnSphere(xs, ys, zs, kdtree, delaunay, _noCells);
		if (voro != NULL) {
			voro->setBoundingBox(dset->boundingBox());
			if (_addCommands)
				_plugins->addCommands(voro);
			_obj->addBasicComponent(voro);

			const std::vector <poca::core::Vec3mf>& normals = ((poca::geometry::VoronoiDiagram2DOnSphere*)voro)->getNormals();
			dset->executeCommand(&poca::core::CommandInfo(false, "addNormals", &normals));

			if (voro != NULL) {
				_obj->notify("LoadObjCharacteristicsAllWidgets");
				_obj->notifyAll("updateDisplay");
			}
		}
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("time for computing Voronoi: %ld ms\n", elapsed);
		return voro;
	}

	VoronoiDiagram* VoronoiDiagramFactory::createVoronoiDiagram(const std::vector <float>& _xs, const std::vector <float>& _ys, const poca::core::BoundingBox& _bbox, KdTree_DetectionPoint* _kdtree, DelaunayTriangulationInterface* _delau)
	{
		DelaunayTriangulation2DDelaunator* d2D = static_cast <DelaunayTriangulation2DDelaunator*>(_delau);

		float w = _bbox[3] - _bbox[0], h = _bbox[4] - _bbox[1], smallestD = w < h ? w : h, epsilon = smallestD * 0.01;
		voronator::Voronator v(d2D->getDelaunator(), _bbox[0] - 1, _bbox[1] - 1, _bbox[3] + 1, _bbox[4] + 1);

		for (std::size_t n = 0; n < v.firsts.size(); n++)
			v.firsts[n] /= 2;
		const std::vector <bool>& borderCells = v.getBorderCells();

		poca::geometry::VoronoiDiagram2D* voro = new poca::geometry::VoronoiDiagram2D(v.getCellPoints(), v.nbEdges(), v.firsts, v.neighs, borderCells, _xs.data(), _ys.data(), NULL, _kdtree, _delau);
		return voro;
	}

	VoronoiDiagram* VoronoiDiagramFactory::createVoronoiDiagram(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector <float>& _zs, KdTree_DetectionPoint* _kdtree, DelaunayTriangulationInterface* _delau, const bool _noCells)
	{
#ifndef NO_CUDA
		int devCount; // Number of CUDA devices
		cudaError_t err = cudaGetDeviceCount(&devCount);
		if (err != cudaSuccess) {
			std::cout << "nope" << std::endl;
			QMessageBox msgBox;
			msgBox.setText("Error: CUDA is not installed, Voronoi 3D is therefore not available.");
			msgBox.exec();
			return NULL;
		}

		DelaunayTriangulation3D* d3D = static_cast <DelaunayTriangulation3D*>(_delau);
		Triangulation_3_inexact* delaunay3D = d3D->getDelaunay();
		uint32_t nbCells = _xs.size();
		clock_t t1, t2;
		t1 = clock();
		assert(delaunay3D->is_valid());

		std::vector <Finite_vertices_iterator_3_inexact> delaunayVertices(nbCells);
		for (Finite_vertices_iterator_3_inexact vit = delaunay3D->finite_vertices_begin(); vit != delaunay3D->finite_vertices_end(); vit++)
			delaunayVertices[vit->info()] = vit;

		double nbPs = delaunayVertices.size();
		unsigned int nbForUpdate = nbPs / 100., cptTimer = 0;
		if (nbForUpdate == 0) nbForUpdate = 1;
		std::printf("3D Voronoi CGAL: %.2f %%", (0. / nbPs * 100.));

		std::vector <uint32_t> neighbors;
		std::vector <uint32_t> indexFirstNeighborCell;
		std::vector <poca::core::Vec3mf> triangles;
		std::vector <uint32_t> firstTriangleCell;
		std::vector <float> cells;
		std::vector <uint32_t> nbTriangleCells;

		uint32_t estimationNbNeighs = 25;
		neighbors.resize(estimationNbNeighs * delaunayVertices.size(), 0);
		indexFirstNeighborCell.resize(delaunayVertices.size() + 1, 0);

		std::vector <uint32_t> columnIndices(estimationNbNeighs * delaunayVertices.size()), rowOffsets(delaunayVertices.size() + 1);
		uint32_t cpt = 0;
		indexFirstNeighborCell[0] = cpt;
		for (size_t i = 0; i < delaunayVertices.size(); i++) {
			Vertex_handle_3_inexact v = delaunayVertices[i];
			rowOffsets[i] = cpt;
			if (v != NULL) {
				std::list<Vertex_handle_3_inexact> neighborsVertices;
				delaunay3D->finite_adjacent_vertices(v, std::back_inserter(neighborsVertices));
				if (cpt + neighborsVertices.size() >= neighbors.size()) {
					std::cout << "Resizing arrays" << std::endl;
					neighbors.resize(neighbors.size() * 2);
					columnIndices.resize(columnIndices.size() * 2);
				}
				for (std::list<Vertex_handle_3_inexact>::iterator it = neighborsVertices.begin(); it != neighborsVertices.end(); it++) {
					Vertex_handle_3_inexact other = *it;
					int indexOther = other->info();
					neighbors[cpt] = indexOther;
					columnIndices[cpt++] = indexOther;
				}
			}
			indexFirstNeighborCell[i + 1] = cpt;
			if (cptTimer++ % nbForUpdate == 0) 	std::printf("\r3D Voronoi CGAL: %.2f %%", (cptTimer / nbPs * 100.));
		}
		rowOffsets[delaunayVertices.size()] = cpt;
		neighbors.resize(cpt);
		neighbors.shrink_to_fit();
		columnIndices.resize(cpt);
		columnIndices.shrink_to_fit();

		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("\nTime for preparing data for Voronoi 3D CUDA: %ld ms\n", elapsed);
		t1 = clock();

		std::vector <float> pts(nbCells * 3);
		for (unsigned int n = 0; n < nbCells; n++) {
			pts[n * 3] = _xs[n];
			pts[n * 3 + 1] = _ys[n];
			pts[n * 3 + 2] = _zs[n];
		}

		bool noConstructionCells = _noCells;
		std::vector <float> volumeTetrahedra, volumeCUDA(delaunayVertices.size(), 0);
		std::vector <uint8_t> borders(delaunayVertices.size(), 0);
		triangles.clear();
		firstTriangleCell.resize(nbCells + 1, 0);
		if (!noConstructionCells) {
			cells.resize(_MAX_T_ * 3 * nbCells);
			nbTriangleCells.resize(nbCells);
		}
		else {
			cells.clear();
			nbTriangleCells.clear();
		}

		//Compute size in Ko needed by m_cells
		unsigned int memoryInGPUNeeded = (_MAX_T_ * 3 * nbCells * sizeof(float)) / 1048576;
		//Determine memory available in GPU
		GLint cur_avail_mem_kb = 0;
		glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);
		cur_avail_mem_kb /= 1024;
		//std::cout << "Available GPU memory: " << cur_avail_mem_kb << std::endl;
		//If array is bigger than available memory in GPU, we need to cut the GPU process in several pieces
		unsigned int nbPieces = 1;
		while (cur_avail_mem_kb < (memoryInGPUNeeded / nbPieces)) nbPieces++;

		computeVoronoiFirstRing(pts, columnIndices, rowOffsets, volumeCUDA, borders, cells, nbTriangleCells, noConstructionCells, ceil((float)cells.size() / (float)nbPieces));

		float maxVolume = -FLT_MAX;
		uint32_t cptT = 0, cptNotConstructed = 0;
		if (!noConstructionCells) {
			cptTimer = 0;
			firstTriangleCell[0] = cptT;
			for (uint32_t i = 0; i < nbCells; i++) {
				if (volumeCUDA[i] > maxVolume)
					maxVolume = volumeCUDA[i];

				uint32_t index = i * _MAX_T_ * 3, tmp = 0;
				std::vector <Point_3_inexact> polyVerticesInexact;
				for (uint32_t j = 0; j < nbTriangleCells[i]; j++) {
					uint32_t index2 = index + j * 3;
					polyVerticesInexact.push_back(Point_3_inexact(cells[index2], cells[index2 + 1], cells[index2 + 2]));
				}
				bool is_nan = !(volumeCUDA[i] == volumeCUDA[i]);
				if (polyVerticesInexact.size() > 3 && !is_nan) {
					Polyhedron_3_inexact poly;
					CGAL::convex_hull_3(polyVerticesInexact.begin(), polyVerticesInexact.end(), poly);
					for (Polyhedron_3_inexact::Facet_const_iterator fi = poly.facets_begin(); fi != poly.facets_end(); fi++) {
						Polyhedron_3_inexact::Halfedge_around_facet_const_circulator hfc = fi->facet_begin();
						poca::core::Vec3mf prec;
						bool firstDone = false;
						do {
							Polyhedron_3_inexact::Halfedge_const_handle hh = hfc;
							Polyhedron_3_inexact::Vertex_const_handle v = hh->vertex();
							triangles.push_back(poca::core::Vec3mf(CGAL::to_double(v->point().x()), CGAL::to_double(v->point().y()), CGAL::to_double(v->point().z())));
							cptT++;
						} while (++hfc != fi->facet_begin());
					}
				}
				else
					cptNotConstructed++;

				firstTriangleCell[i + 1] = cptT;
				if (cptTimer++ % nbForUpdate == 0) 	std::printf("\r3D Voronoi CGAL - cell construction: %.2f %%", (cptTimer / nbPs * 100.));
			}

			std::cout << std::endl << "Nb cells not constructed = " << cptNotConstructed << std::endl;
			clock_t t3 = clock();
			elapsed = ((double)t3 - t1) / CLOCKS_PER_SEC * 1000;
			printf("time for constructing cells: %ld ms\n", elapsed);
		}
		else {
			std::vector<float>::iterator itmax = std::max_element(volumeCUDA.begin(), volumeCUDA.end());
			maxVolume = *itmax;
		}

		for (size_t n = 0; n < nbCells; n++)
			if (volumeCUDA[n] <= 0.f || (!(volumeCUDA[n] == volumeCUDA[n])))
				volumeCUDA[n] = maxVolume;

		t2 = clock();
		elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("time for computing Voronoi 3D CUDA: %ld ms\n", elapsed);

		std::vector <bool> bordersBool(borders.size());
		std::transform(borders.begin(), borders.end(), bordersBool.begin(), [](uint8_t x) { return x != 0; });

		poca::geometry::VoronoiDiagram3D* voro = NULL;
		if(!noConstructionCells)
			voro = new poca::geometry::VoronoiDiagram3D(nbCells, neighbors, indexFirstNeighborCell, triangles, firstTriangleCell, volumeCUDA, bordersBool, _xs.data(), _ys.data(), _zs.data(), _kdtree, _delau);
		else
			voro = new poca::geometry::VoronoiDiagram3D(nbCells, neighbors, indexFirstNeighborCell, volumeCUDA, bordersBool, _xs.data(), _ys.data(), _zs.data(), _kdtree, _delau);
		return voro;
#else
	QMessageBox msgBox;
	msgBox.setText("Computing 3D Voronoi diagrams requires an NVidia graphics card and CUDA installed.");
	msgBox.exec();
	return NULL;
#endif
	}

	VoronoiDiagram* VoronoiDiagramFactory::createVoronoiDiagramOnSphere(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector <float>& _zs, KdTree_DetectionPoint* _kdtree, DelaunayTriangulationInterface* _delau, const bool _noCells)
	{
		DelaunayTriangulation2DOnSphere* delaunay = dynamic_cast <DelaunayTriangulation2DOnSphere*>(_delau);
		if (delaunay == NULL) return NULL;

		CGALDelaunayOnSphere* delau = delaunay->getDelaunay();
		const poca::core::Vec3mf& centroid = delaunay->getCentroid();
		const float radius = delaunay->getRadius();

		std::map < CGALDelaunayOnSphere::Vertex_handle, uint32_t > links;
		uint32_t cpt = 0, n = 0;
		for (CGALDelaunayOnSphere::Finite_vertices_iterator it = delau->finite_vertices_begin(); it != delau->finite_vertices_end(); it++, cpt++) {
			CGALDelaunayOnSphere::Vertex_handle v = it;
			links[v] = cpt;
		}

		std::vector <poca::core::Vec3mf> normals(cpt);
		std::vector <poca::core::Vec3md> cellPoints;
		std::vector<uint32_t> firsts(cpt + 1), neighs;
		firsts[0] = 0;

		for (CGALDelaunayOnSphere::Finite_vertices_iterator it = delau->finite_vertices_begin(); it != delau->finite_vertices_end(); it++, n++) {
			poca::core::Vec3mf vertex(it->point().x(), it->point().y(), it->point().z()), vector(vertex - centroid);
			vector.normalize();
			normals[n] = vector;
			CGALDelaunayOnSphere::Edge_circulator first = delau->incident_edges(it), current = first;
			do {
				CGALDelaunayOnSphere::Vertex_handle neigh = current->first->vertex(current->first->ccw(current->second));
				SphereSegment_3 dual = delau->dual(current);
				poca::core::Vec3md p1(dual.source().x(), dual.source().y(), dual.source().z()), p2(dual.target().x(), dual.target().y(), dual.target().z());
				poca::core::Vec3md v1(p1 - vertex), v2(p2 - vertex);
				v1.normalize();
				v2.normalize();
				poca::core::Vec3md cross = v1.cross(v2);
				cellPoints.push_back(poca::core::Vec3md(p2));
				neighs.push_back(links[neigh]);//cellPoints.size() - 1);
				firsts[n + 1] = neighs.size();
				current++;
			} while (current != first);
		}

		poca::geometry::VoronoiDiagram2DOnSphere* voro = new poca::geometry::VoronoiDiagram2DOnSphere(cellPoints, cellPoints.size(), firsts, neighs, normals, _xs.data(), _ys.data(), _zs.data(), centroid, radius, _kdtree, _delau);
		return voro;
	}
}

