#include <Windows.h>
#include <GL/glew.h>
#include <omp.h>
#include <unordered_set>
#include <algorithm>
#include <fstream>

#include <General/Vec3.hpp>
#include <General/MyArray.hpp>

#include "VoronoiDiagramFactory.hpp"
#include "../Geometry/voronator.hpp"
#include "../General/CGAL_includes.hpp"
#include "../3D_voronoi_GPU/voronoi.h"

#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

namespace poca::geometry {
	VoronoiDiagramFactory::VoronoiDiagramFactory()
	{

	}

	VoronoiDiagramFactory::~VoronoiDiagramFactory()
	{

	}

	VoronoiDiagram* VoronoiDiagramFactory::createVoronoiDiagram(const std::vector <float>& _xs, const std::vector <float>& _ys, const poca::core::BoundingBox& _bbox, KdTree_DetectionPoint* _kdtree, DelaunayTriangulationInterface* _delau)
	{
		std::vector <double> coords;
		coords.resize(_xs.size() * 2);
		for (size_t n = 0; n < _xs.size(); n++) {
			coords[2 * n] = _xs[n];
			coords[2 * n + 1] = _ys[n];
		}
		
		float w = _bbox[3] - _bbox[0], h = _bbox[4] - _bbox[1], smallestD = w < h ? w : h, epsilon = smallestD * 0.01;
		poca::core::BoundingBox bbox(_bbox[0] - epsilon, _bbox[1] - epsilon, 0.f, _bbox[3] + epsilon, _bbox[4] + epsilon, 0.f);

		delaunator::Delaunator d(coords);
		delaunator::DelaunayFromDelaunator delau(d);

		voronator::Voronator v(delau, _bbox[0] - 1, _bbox[1] - 1, _bbox[3] + 1, _bbox[4] + 1);

		for (std::size_t n = 0; n < v.firsts.size(); n++)
			v.firsts[n] /= 2;

		poca::geometry::VoronoiDiagram2D* voro = new poca::geometry::VoronoiDiagram2D(v.getCellPoints(), v.nbEdges(), v.firsts, v.neighs, _xs.data(), _ys.data(), _kdtree, _delau);
		return voro;
	}

	VoronoiDiagram* VoronoiDiagramFactory::createVoronoiDiagram(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector <float>& _zs, KdTree_DetectionPoint* _kdtree, DelaunayTriangulationInterface* _delau, const bool _noCells)
	{
		size_t nbCells = _xs.size();
		clock_t t1, t2;
		double xmin, xmax, ymin, ymax, zmin, zmax;
		xmin = ymin = zmin = DBL_MAX;
		xmax = ymax = zmax = -DBL_MAX;
		for (unsigned int n = 0; n < nbCells; n++) {
			xmin = (_xs[n] < xmin) ? _xs[n] : xmin;
			xmax = (_xs[n] > xmax) ? _xs[n] : xmax;
			ymin = (_ys[n] < ymin) ? _ys[n] : ymin;
			ymax = (_ys[n] > ymax) ? _ys[n] : ymax;
			zmin = (_zs[n] < zmin) ? _zs[n] : zmin;
			zmax = (_zs[n] > zmax) ? _zs[n] : zmax;
		}

		float w = xmax - xmin, h = ymax - ymin, t = zmax - zmin, smallestD = w < h ? w : h;
		smallestD = smallestD < t ? smallestD : t;
		float epsilon = smallestD * 0.01;
		poca::core::BoundingBox bbox(xmin - epsilon, ymin - epsilon, zmin - epsilon, xmax + epsilon, ymax + epsilon, zmax + epsilon);

		t1 = clock();
		std::vector < std::pair< Point_delau_3D_inexact, int > > V;
		V.reserve(nbCells);
		for (int i = 0; i != nbCells; ++i)
			V.push_back(std::make_pair(Point_delau_3D_inexact(_xs[i], _ys[i], _zs[i]), i));

		// Construct the locking data-structure, using the bounding-box of the points
		Triangulation_3_inexact::Lock_data_structure locking_ds(CGAL::Bbox_3(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]), 50);
		// Construct the triangulation in parallel
		Triangulation_3_inexact delaunay3D(V.begin(), V.end(), &locking_ds);
		std::cout << "# of vertices = " << delaunay3D.number_of_vertices() << std::endl;
		std::cout << "# of cells = " << delaunay3D.number_of_cells() << std::endl;
		std::cout << "# of finite cells = " << delaunay3D.number_of_finite_cells() << std::endl;
		assert(delaunay3D.is_valid());

		for (Triangulation_3_inexact::All_cells_iterator cit = delaunay3D.all_cells_begin(); cit != delaunay3D.all_cells_end(); cit++)
			cit->info() = -1;

		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("time for computing 3D Delaunay CGAL: %ld ms\n", elapsed);

		std::vector <Finite_vertices_iterator_3_inexact> delaunayVertices(nbCells);
		for (Finite_vertices_iterator_3_inexact vit = delaunay3D.finite_vertices_begin(); vit != delaunay3D.finite_vertices_end(); vit++)
			delaunayVertices[vit->info()] = vit;

		double nbPs = delaunayVertices.size();
		unsigned int nbForUpdate = nbPs / 100., cptTimer = 0;
		if (nbForUpdate == 0) nbForUpdate = 1;
		std::printf("3D Voronoi CGAL: %.2f %%", (0. / nbPs * 100.));

		std::vector <int> neighbors;
		std::vector <size_t> indexFirstNeighborCell;
		std::vector <poca::core::Vec3mf> triangles;
		std::vector <unsigned int> firstTriangleCell;
		std::vector <float> cells;
		std::vector <unsigned int> nbTriangleCells;

		neighbors.resize(500 * delaunayVertices.size(), 0);
		indexFirstNeighborCell.resize(delaunayVertices.size() + 1, 0);

		std::vector <unsigned int> columnIndices(/*nbNeighsTotal*/500 * delaunayVertices.size()), rowOffsets(delaunayVertices.size() + 1);
		size_t cpt = 0;
		indexFirstNeighborCell[0] = cpt;
		for (size_t i = 0; i < delaunayVertices.size(); i++) {
			Vertex_handle_3_inexact v = delaunayVertices[i];
			if (v == NULL) continue;
			rowOffsets[i] = cpt;
			std::list<Vertex_handle_3_inexact> neighborsVertices;
			delaunay3D.finite_adjacent_vertices(v, std::back_inserter(neighborsVertices));
			for (std::list<Vertex_handle_3_inexact>::iterator it = neighborsVertices.begin(); it != neighborsVertices.end(); it++) {
				Vertex_handle_3_inexact other = *it;
				int indexOther = other->info();
				neighbors[cpt] = indexOther;
				columnIndices[cpt++] = indexOther;
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
		elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("\ntime for preparing data for Voronoi 3D CUDA: %ld ms\n", elapsed);
		t1 = clock();

		std::vector <float> pts(nbCells * 3);
		for (unsigned int n = 0; n < nbCells; n++) {
			pts[n * 3] = _xs[n];
			pts[n * 3 + 1] = _ys[n];
			pts[n * 3 + 2] = _zs[n];
		}

		bool noConstructionCells = _noCells;
		std::vector <float> volumeTetrahedra, volumeCUDA;
		volumeCUDA.resize(delaunayVertices.size(), 0);
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
		std::cout << "Available GPU memory: " << cur_avail_mem_kb << std::endl;
		//If array is bigger than available memory in GPU, we need to cut the GPU process in several pieces
		unsigned int nbPieces = 1;
		while (cur_avail_mem_kb < (memoryInGPUNeeded / nbPieces)) nbPieces++;

		computeVoronoiFirstRing(pts, columnIndices, rowOffsets, volumeCUDA, cells, nbTriangleCells, noConstructionCells, ceil((float)cells.size() / (float)nbPieces));

		float maxVolume = -FLT_MAX;
		unsigned int cptT = 0, cptNotConstructed = 0;
		if (!noConstructionCells) {
			cptTimer = 0;
			firstTriangleCell[0] = cptT;
			for (unsigned int i = 0; i < nbCells; i++) {
				if (volumeCUDA[i] > maxVolume)
					maxVolume = volumeCUDA[i];

				unsigned int index = i * _MAX_T_ * 3, tmp = 0;
				std::vector <Point_3_inexact> polyVerticesInexact;
				for (unsigned int j = 0; j < nbTriangleCells[i]; j++) {
					unsigned int index2 = index + j * 3;
					polyVerticesInexact.push_back(Point_3_inexact(cells[index2], cells[index2 + 1], cells[index2 + 2]));
				}
				bool is_nan = !(volumeCUDA[i] == volumeCUDA[i]);
				if (polyVerticesInexact.size() > 3 && !is_nan) {// (is_nan || m_volumeCUDA[i] > 0)){
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
		poca::geometry::VoronoiDiagram3D* voro = NULL;
		if(!noConstructionCells)
			voro = new poca::geometry::VoronoiDiagram3D(nbCells, neighbors, indexFirstNeighborCell, triangles, firstTriangleCell, volumeCUDA, _xs.data(), _ys.data(), _zs.data(), _kdtree, _delau);
		else
			voro = new poca::geometry::VoronoiDiagram3D(nbCells, neighbors, indexFirstNeighborCell, volumeCUDA, _xs.data(), _ys.data(), _zs.data(), _kdtree, _delau);
		return voro;
	}
}