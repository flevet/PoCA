#include <fstream>
#include <limits>

#include "DelaunayTriangulationFactory.hpp"
#include "../Geometry/delaunator.hpp"
#include "../General/CGAL_includes.hpp"

namespace poca::geometry {
	DelaunayTriangulationFactory::DelaunayTriangulationFactory()
	{

	}

	DelaunayTriangulationFactory::~DelaunayTriangulationFactory()
	{

	}

	DelaunayTriangulation* DelaunayTriangulationFactory::createDelaunayTriangulation(const std::vector <float>& _xs, const std::vector <float>& _ys)
	{
		std::vector <double> coords;
		coords.resize(_xs.size() * 2);
		for (size_t n = 0; n < _xs.size(); n++) {
			coords[2 * n] = _xs[n];
			coords[2 * n + 1] = _ys[n];
		}
		delaunator::Delaunator d(coords);
		delaunator::DelaunayFromDelaunator delau(d);
		size_t nbTriangles = delau.nbTriangles();

		std::vector <std::size_t> neighbors, firsts(nbTriangles + 1), currentNeighs(3);
		firsts[0] = 0;
		for (size_t n = 0; n < nbTriangles; n++) {
			delau.trianglesAdjacentToTriangle(n, currentNeighs);
			firsts[n + 1] = firsts[n] + currentNeighs.size();
			std::copy(currentNeighs.begin(), currentNeighs.end(), std::back_inserter(neighbors));
		}
		//for (std::size_t n = 0; n < neighbors.size(); n++)
		//	neighbors[n] /= 2;
		poca::core::MyArraySizeT neighs;
		neighs.initialize(neighbors, firsts);

		return new DelaunayTriangulation2D(_xs, _ys, d.triangles, neighs);
	}

	DelaunayTriangulation* DelaunayTriangulationFactory::createDelaunayTriangulation(const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector <float>& _zs)
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
		std::vector < std::pair< Point_delau_3D_inexact, int > > V;
		V.reserve(_xs.size());
		for (int i = 0; i != _xs.size(); ++i)
			V.push_back(std::make_pair(Point_delau_3D_inexact(_xs[i], _ys[i], _zs[i]), i));

		// Construct the locking data-structure, using the bounding-box of the points
		Triangulation_3_inexact::Lock_data_structure locking_ds(CGAL::Bbox_3(xmin, ymin, zmin, xmax, ymax, zmax), 50);
		// Construct the triangulation in parallel
		Triangulation_3_inexact delaunay3D(V.begin(), V.end(), &locking_ds);
		std::cout << "# of vertices = " << delaunay3D.number_of_vertices() << std::endl;
		std::cout << "# of cells = " << delaunay3D.number_of_cells() << std::endl;
		std::cout << "# of finite cells = " << delaunay3D.number_of_finite_cells() << std::endl;
		assert(delaunay3D.is_valid());

		size_t nbFiniteCells = (size_t)delaunay3D.number_of_finite_cells(), cpt = 0, cptT = 0, cptV = 0;

		std::vector < Cell_handle_3_inexact> handles(nbFiniteCells);
		Finite_cells_iterator_3_inexact fit = delaunay3D.finite_cells_begin();
		Finite_cells_iterator_3_inexact fdone = delaunay3D.finite_cells_end();
		for (; fit != fdone; fit++) {
			Cell_handle_3_inexact f = fit;
			f->info() = cpt++;
			handles[f->info()] = f;
		}
		cpt = 0;

		//std::ofstream fs("e:/vol_opensmlm.txt");
		//std::vector <poca::core::Vec3mf> triangles(nbFiniteCells * 4 * 3);//4 triangles per cell, 3 coords per triangle
		std::vector <size_t> triangles(nbFiniteCells * 4 * 3);//4 triangles per cell, 3 coords per triangle
		std::vector <size_t> neighbors(handles.size() * 4), firsts(nbFiniteCells + 1);
		std::vector <float> volumes(nbFiniteCells);
		firsts[0] = 0;
		for (size_t n = 0; n < handles.size(); n++) {
			Cell_handle_3_inexact c = handles[n];
			for (int i = 0; i < 4; i++) {
				Cell_handle_3_inexact other = c->neighbor(i);
				neighbors[cpt++] = delaunay3D.is_infinite(other) ? std::numeric_limits<std::size_t>::max() : other->info();

				unsigned int cur = i, next = (i + 1) % 4, nnext = (i + 2) % 4;
				triangles[cptT++] = c->vertex(cur)->info();// .set(c->vertex(cur)->point().x(), c->vertex(cur)->point().y(), c->vertex(cur)->point().z());
				triangles[cptT++] = c->vertex(next)->info();//.set(c->vertex(next)->point().x(), c->vertex(next)->point().y(), c->vertex(next)->point().z());
				triangles[cptT++] = c->vertex(nnext)->info();//.set(c->vertex(nnext)->point().x(), c->vertex(nnext)->point().y(), c->vertex(nnext)->point().z());
				//triangles[cptT++].set(c->vertex(cur)->point().x(), c->vertex(cur)->point().y(), c->vertex(cur)->point().z());
				//triangles[cptT++].set(c->vertex(next)->point().x(), c->vertex(next)->point().y(), c->vertex(next)->point().z());
				//triangles[cptT++].set(c->vertex(nnext)->point().x(), c->vertex(nnext)->point().y(), c->vertex(nnext)->point().z());
			}
			Triangulation_3_inexact::Tetrahedron tetr = delaunay3D.tetrahedron(c);
			volumes[cptV++] = tetr.volume();
			//fs << volumes[cptV - 1] << std::endl;
			firsts[n + 1] = cpt;
		}
		poca::core::MyArraySizeT neighs;
		neighs.initialize(neighbors, firsts);
		//fs.close();

		return new DelaunayTriangulation3D(_xs, _ys, _zs, triangles, volumes, neighs, nbFiniteCells);
	}
}