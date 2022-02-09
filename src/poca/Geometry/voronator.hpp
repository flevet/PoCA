#pragma once

#include <General/Vec2.hpp>

#include "../Geometry/delaunator.hpp"

namespace voronator {

	class Voronator {
	public:
		delaunator::DelaunayFromDelaunator & delaunay;
		delaunator::Delaunator & delaunator;

		std::vector <double> circumcenters;
		std::vector <double> vectors;

		std::vector <double> cells;
		std::vector <size_t> firsts, neighs;

		double xmin, ymin, xmax, ymax;

		Voronator(delaunator::DelaunayFromDelaunator & _delau, const double _xmin, const double _ymin, const double _xmax, const double _ymax);

		void cell(std::size_t i, std::vector <double> & _points);
		void clip(std::size_t i, std::vector <double> & _points);
		void clipFinite(std::size_t i, std::vector <double> & _points);
		void clipInfinite(std::size_t i, std::vector <double> & _points, const double vx0, const double vy0, const double vxn, const double vyn);
		std::size_t clipSegment(double x0, double y0, double x1, double y1, unsigned char c0, unsigned char c1, double & sx0, double & sy0, double & sx1, double & sy1);
		std::size_t edge(std::size_t i, unsigned char e0, unsigned char e1, std::vector <double> & P, std::size_t j);
		bool contains(std::size_t i, double x, double y);
		void neighbors(std::size_t, std::vector <size_t> &, std::size_t);

		std::size_t project(double x0, double y0, double vx, double vy, double & x, double & y);
		unsigned char edgecode(double x, double y);
		unsigned char regioncode(double x, double y);

		inline poca::core::Vec2md * getCellPoints() const { return (poca::core::Vec2md *)cells.data(); }
		inline unsigned int * getFirstPointCells() const { return (unsigned int *)firsts.data(); }
		inline unsigned int * getIndexNeighbors() const { return (unsigned int *)neighs.data(); }

		inline const int getNSites() const { return (int)(delaunay.points.size() / 2); }
		inline const int nbEdges() const { return (int)neighs.size(); }
	};
}