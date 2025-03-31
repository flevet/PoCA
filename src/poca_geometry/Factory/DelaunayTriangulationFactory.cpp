/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulationFactory.cpp
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
#include <limits>

#include <Interfaces/MyObjectInterface.hpp>
#include <General/Histogram.hpp>
#include <General/PluginList.hpp>

#include "DelaunayTriangulationFactory.hpp"
#include "../Geometry/DelaunayTriangulation.hpp"
#include "../Geometry/delaunator.hpp"
#include "../Geometry/CGAL_includes.hpp"
#include "../Geometry/BasicComputation.hpp"

namespace poca::geometry {
	DelaunayTriangulationFactoryInterface* createDelaunayTriangulationFactory() {
		return new DelaunayTriangulationFactory();
	}

	DelaunayTriangulationFactory::DelaunayTriangulationFactory()
	{

	}

	DelaunayTriangulationFactory::~DelaunayTriangulationFactory()
	{

	}

	DelaunayTriangulationInterface* DelaunayTriangulationFactory::createDelaunayTriangulation(poca::core::MyObjectInterface* _obj, poca::core::PluginList* _plugins, const bool _addCommands)
	{
		clock_t t1, t2;
		t1 = clock();

		poca::geometry::DelaunayTriangulationInterface* delau = nullptr;
		poca::core::BasicComponentInterface* dset = _obj->getBasicComponent("DetectionSet");
		if (dset == NULL)
			return NULL;
		const std::vector <float>& xs = static_cast<poca::core::Histogram<float>*>(dset->getOriginalHistogram("x"))->getValues();
		const std::vector <float>& ys = static_cast<poca::core::Histogram<float>*>(dset->getOriginalHistogram("y"))->getValues();
		if (!dset->hasData("z")) {
			delau = createDelaunayTriangulation(xs, ys);
		}
		else {
			const std::vector <float>& zs = static_cast<poca::core::Histogram<float>*>(dset->getOriginalHistogram("z"))->getValues();
			delau = createDelaunayTriangulation(xs, ys, zs);
		}
		if (delau != NULL) {
			delau->setBoundingBox(dset->boundingBox());
			if(_addCommands)
				_plugins->addCommands(delau);
			_obj->addBasicComponent(delau);
			if (delau != NULL) {
				_obj->notify("LoadObjCharacteristicsAllWidgets");
				_obj->notifyAll("updateDisplay");
			}
		}

		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("time for computing the Delaunay triangulation: %ld ms\n", elapsed);
		return delau;
	}

	DelaunayTriangulationInterface* DelaunayTriangulationFactory::createDelaunayTriangulationOnSphere(poca::core::MyObjectInterface* _obj, poca::core::PluginList* _plugins, const poca::core::Vec3mf& _center, const float _radius, const bool _addCommands)
	{
		//return NULL;

		clock_t t1, t2;
		t1 = clock();

		poca::geometry::DelaunayTriangulationInterface* delau = nullptr;
		poca::core::BasicComponentInterface* dset = _obj->getBasicComponent("DetectionSet");
		if (dset == NULL || !dset->hasData("z"))
			return NULL;
		const std::vector <float>& xs = static_cast<poca::core::Histogram<float>*>(dset->getOriginalHistogram("x"))->getValues();
		const std::vector <float>& ys = static_cast<poca::core::Histogram<float>*>(dset->getOriginalHistogram("y"))->getValues();
		const std::vector <float>& zs = static_cast<poca::core::Histogram<float>*>(dset->getOriginalHistogram("z"))->getValues();

		delau = createDelaunayTriangulationOnSphere(xs, ys, zs, _center, _radius);
		if (delau != NULL) {
			delau->setBoundingBox(dset->boundingBox());
			if (_addCommands)
				_plugins->addCommands(delau);
			_obj->addBasicComponent(delau);
			if (delau != NULL) {
				_obj->notify("LoadObjCharacteristicsDelaunayTriangulationWidget");
				_obj->notifyAll("updateDisplay");
			}
		}

		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("time for computing the Delaunay triangulation: %ld ms\n", elapsed);
		return delau;
	}

	DelaunayTriangulationInterface* DelaunayTriangulationFactory::createDelaunayTriangulation(const std::vector <float>& _xs, const std::vector <float>& _ys)
	{
		std::vector <double> * coords = new std::vector <double>();
		coords->resize(_xs.size() * 2);
		for (size_t n = 0; n < _xs.size(); n++) {
			(*coords)[2 * n] = _xs[n];
			(*coords)[2 * n + 1] = _ys[n];
		}
		delaunator::Delaunator * d = new delaunator::Delaunator(*coords);
		delaunator::DelaunayFromDelaunator* delau = new delaunator::DelaunayFromDelaunator(d);
		size_t nbTriangles = delau->nbTriangles();

		std::vector <std::uint32_t> neighbors, firsts(nbTriangles + 1), currentNeighs(3);
		firsts[0] = 0;
		for (size_t n = 0; n < nbTriangles; n++) {
			delau->trianglesAdjacentToTriangleUint32(n, currentNeighs);
			firsts[n + 1] = firsts[n] + currentNeighs.size();
			std::copy(currentNeighs.begin(), currentNeighs.end(), std::back_inserter(neighbors));
		}

		//test computing area
		/*std::ofstream fs("e:/areas_tri.txt");
		fs << "Side\tPoints\tCGAL" << std::endl;
		for (size_t t = 0; t < nbTriangles; t++) {
			size_t idx[] = { t * 3, t * 3 + 1, t * 3 + 2 };
			double t1x = delau->points[2 * delau->delaunator->triangles[idx[0]]], t1y = delau->points[2 * delau->delaunator->triangles[idx[0]] + 1];
			double t2x = delau->points[2 * delau->delaunator->triangles[idx[1]]], t2y = delau->points[2 * delau->delaunator->triangles[idx[1]] + 1];
			double t3x = delau->points[2 * delau->delaunator->triangles[idx[2]]], t3y = delau->points[2 * delau->delaunator->triangles[idx[2]] + 1];
		
			poca::core::Vec3mf v1(t1x, t1y, 0), v2(t2x, t2y, 0), v3(t3x, t3y, 0);

			float sideA = (v1 - v2).length(), sideB = (v1 - v3).length(), sideC = (v2 - v3).length();
			float areaSide = poca::geometry::computeAreaTriangle<float>(sideA, sideB, sideC);

			float areaPoint = poca::geometry::computeTriangleArea(v1.x(), v1.y(), v2.x(), v2.y(), v3.x(), v3.y());

			Point_2 p1(v1.x(), v1.y()), p2(v2.x(), v2.y()), p3(v3.x(), v3.y());
			float areaCGAL = CGAL::area(p1, p2, p3);

			fs << areaSide << "\t" << areaPoint << "\t" << areaCGAL << std::endl;
		}
		fs.close();*/

		poca::core::MyArrayUInt32 neighs;
		neighs.initialize(neighbors, firsts);

		return new DelaunayTriangulation2DDelaunator(_xs, _ys, d->triangles, neighs, delau, coords);
	}

	DelaunayTriangulationInterface* DelaunayTriangulationFactory::createDelaunayTriangulation(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector <float>& _zs)
	{
		clock_t t1, t2;
		double xmin, xmax, ymin, ymax, zmin, zmax;
		xmin = ymin = zmin = DBL_MAX;
		xmax = ymax = zmax = -DBL_MAX;
		for (unsigned int n = 0; n < _xs.size(); n++) {
			xmin = (_xs[n] < xmin) ? _xs[n] : xmin;
			xmax = (_xs[n] > xmax) ? _xs[n] : xmax;
			ymin = (_ys[n] < ymin) ? _ys[n] : ymin;
			ymax = (_ys[n] > ymax) ? _ys[n] : ymax;
			zmin = (_zs[n] < zmin) ? _zs[n] : zmin;
			zmax = (_zs[n] > zmax) ? _zs[n] : zmax;
		}

		t1 = clock();
		std::vector < std::pair< Point_delau_3D_inexact, uint32_t > > V;
		V.reserve(_xs.size());
		for (size_t i = 0; i != _xs.size(); ++i) {
			V.push_back(std::make_pair(Point_delau_3D_inexact(_xs[i], _ys[i], _zs[i]), i));
		}

		// Construct the locking data-structure, using the bounding-box of the points
		Triangulation_3_inexact::Lock_data_structure locking_ds(CGAL::Bbox_3(xmin, ymin, zmin, xmax, ymax, zmax), 50);
		// Construct the triangulation in parallel
		Triangulation_3_inexact* delaunay3D = new Triangulation_3_inexact(V.begin(), V.end(), &locking_ds);
		
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		std::cout << "Delaunay triangulation 3D created in " << elapsed << ", composed of : " << std::endl;
		std::cout << "		----> " << delaunay3D->number_of_vertices() << " vertices" << std::endl;
		std::cout << "		----> " << delaunay3D->number_of_finite_cells() << " tetrahedra" << std::endl;
		assert(delaunay3D->is_valid());
		uint32_t nbFiniteCells = (uint32_t)delaunay3D->number_of_finite_cells();

		for (Triangulation_3_inexact::All_cells_iterator cit = delaunay3D->all_cells_begin(); cit != delaunay3D->all_cells_end(); cit++) {
			cit->info() = -1;
		}

		//sort triangles by centroid order
		uint32_t cpttt = 0;
		std::vector < Cell_handle_3_inexact> tetraNoOrder(nbFiniteCells);
		std::vector <poca::core::Vec3mf> centroids(nbFiniteCells, poca::core::Vec3mf(0.f, 0.f, 0.f));
		for (Triangulation_3_inexact::Finite_cells_iterator cit = delaunay3D->finite_cells_begin(); cit != delaunay3D->finite_cells_end(); cit++, cpttt++) {
			for (int i = 0; i < 4; i++) {
				Vertex_handle_3_inexact v = cit->vertex(i);
				centroids[cpttt] += poca::core::Vec3mf(v->point().x() / 4.f, v->point().y() / 4.f, v->point().z() / 4.f);
				tetraNoOrder[cpttt] = cit;
			}
		}
		// initialize original index locations
		std::vector<size_t> idx(centroids.size());
		iota(idx.begin(), idx.end(), 0);
		// sort indexes based on comparing values in v
		// using std::stable_sort instead of std::sort
		// to avoid unnecessary index re-orderings
		// when v contains elements of equal values 
		std::stable_sort(idx.begin(), idx.end(), [&centroids](size_t i1, size_t i2) {return centroids[i1] < centroids[i2]; });

		uint32_t cpt = 0, cptT = 0, cptV = 0;

		std::vector < Cell_handle_3_inexact> handles(nbFiniteCells);

		for (auto n = 0; n < idx.size(); n++) {
			Cell_handle_3_inexact f = tetraNoOrder[idx[n]];
			f->info() = n;
			handles[f->info()] = f;
		}

		cpt = 0;

		std::vector <uint32_t> triangles(nbFiniteCells * 4 * 3);//4 triangles per cell, 3 coords per triangle
		std::vector <uint32_t> neighbors(handles.size() * 4), firsts(nbFiniteCells + 1);
		std::vector <float> volumes(nbFiniteCells);
		firsts[0] = 0;
		double nbPs = handles.size();
		unsigned int nbForUpdate = nbPs / 100.;
		if (nbForUpdate == 0) nbForUpdate = 1;
		printf("\rComputing tetrahedrons volume: %.2f %%", (0. / nbPs * 100.));
		for (size_t n = 0; n < handles.size(); n++) {
			if (n % nbForUpdate == 0) printf("\rComputing tetrahedrons volume: %.2f %%", ((double)n / nbPs * 100.));
			Cell_handle_3_inexact c = handles[n];
			for (int i = 0; i < 4; i++) {
				Cell_handle_3_inexact other = c->neighbor(i);
				neighbors[cpt++] = delaunay3D->is_infinite(other) ? std::numeric_limits<std::uint32_t>::max() : other->info();

				unsigned int cur = i, next = (i + 1) % 4, nnext = (i + 2) % 4;
				triangles[cptT++] = c->vertex(cur)->info();
				triangles[cptT++] = c->vertex(next)->info();
				triangles[cptT++] = c->vertex(nnext)->info();
			}
			Triangulation_3_inexact::Tetrahedron tetr = delaunay3D->tetrahedron(c);
			volumes[cptV++] = tetr.volume();
			firsts[n + 1] = cpt;
		}
		printf("\rComputing tetrahedrons volume: 100.00 %%\n");
		poca::core::MyArrayUInt32 neighs;
		neighs.initialize(neighbors, firsts);

		DelaunayTriangulation3D* d3D = new DelaunayTriangulation3D(_xs, _ys, _zs, triangles, volumes, neighs, nbFiniteCells, delaunay3D);
		return d3D;
	}

	DelaunayTriangulationInterface* DelaunayTriangulationFactory::createDelaunayTriangulationOnSphere(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector <float>& _zs, const poca::core::Vec3mf& _center, const float _radius)
	{
		std::vector <float> xsproj(_xs.size()), ysproj(_xs.size()), zsproj(_xs.size());
		for (size_t n = 0; n < xsproj.size(); n++) {
			poca::core::Vec3mf v(_xs[n] - _center.x(), _ys[n] - _center.y(), _zs[n] - _center.z());
			v.normalize();
			poca::core::Vec3mf pt(_center + _radius * v);
			xsproj[n] = pt.x();
			ysproj[n] = pt.y();
			zsproj[n] = pt.z();
		}

		SphereTraits traits(SpherePoint_3(_center.x(), _center.y(), _center.z()), _radius);
		CGALDelaunayOnSphere* delaunay = new CGALDelaunayOnSphere(traits);
		SphereTraits::Construct_point_on_sphere_2 cst = traits.construct_point_on_sphere_2_object();
		for (size_t n = 0; n < xsproj.size(); n++) {
			SpherePoint_3 pt(xsproj[n], ysproj[n], zsproj[n]);
			delaunay->insert(cst(pt));
		}

		std::cout << delaunay->number_of_vertices() << " vertices" << std::endl;
		std::cout << delaunay->number_of_faces() << " solid faces" << std::endl;

		uint32_t nbTriangles = 0;
		size_t cpt = 0;
		//First, we need to identify the points by their index -> construct a vector of pair vertex handle to index
		std::vector <std::pair<CGALDelaunayOnSphere::Vertex_handle, size_t>> pairsVertex;
		for (auto it = delaunay->all_vertices_begin(); it != delaunay->all_vertices_end(); it++) {
			pairsVertex.push_back(std::make_pair(it, cpt++));
		}

		std::vector <size_t> triangles;
		std::vector <std::pair<CGALDelaunayOnSphere::Face_handle, uint32_t>> pairs;
		for (auto it = delaunay->finite_faces_begin(); it != delaunay->finite_faces_end(); it++) {
			pairs.push_back(std::make_pair(it, nbTriangles++));
			for (int i = 2; i >= 0; i--) {
				auto v = it->vertex(i);
				auto it = std::find_if(pairsVertex.begin(), pairsVertex.end(), [&v](const std::pair<CGALDelaunayOnSphere::Vertex_handle, uint32_t>& element) { return element.first == v; });
				triangles.push_back(it->second);
			}
		}

		std::vector <std::uint32_t> neighbors, firsts(nbTriangles + 1), currentNeighs(3);
		firsts[0] = 0;
		size_t n = 0;
		for (auto triangle = delaunay->finite_faces_begin(); triangle != delaunay->finite_faces_end(); triangle++, n++) {
			for (int i = 2, j = 0; i >= 0; i--, j++) {
				auto neighbor = triangle->neighbor(i);
				auto it = std::find_if(pairs.begin(), pairs.end(), [&neighbor](const std::pair<CGALDelaunayOnSphere::Face_handle, uint32_t>& element) { return element.first == neighbor; });
				currentNeighs[j] = it->second;
			}
			firsts[n + 1] = firsts[n] + currentNeighs.size();
			std::copy(currentNeighs.begin(), currentNeighs.end(), std::back_inserter(neighbors));
		}

		poca::core::MyArrayUInt32 neighs;
		neighs.initialize(neighbors, firsts);

		return new DelaunayTriangulation2DOnSphere(xsproj, ysproj, zsproj, triangles, neighs, delaunay, _center, _radius);
	}
}

