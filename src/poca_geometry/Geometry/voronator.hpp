/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      voronator.hpp
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

#pragma once

#include <General/Vec2.hpp>
#include <General/Vec3.hpp>

#include "../Geometry/delaunator.hpp"

namespace voronator {

	class Voronator {
	public:
		delaunator::DelaunayFromDelaunator * delaunay;
		delaunator::Delaunator * delaunator;

		std::vector <double> circumcenters;
		std::vector <double> vectors;

		std::vector <double> cells;
		std::vector <uint32_t> firsts, neighs;

		std::vector <bool> borderCells;

		double xmin, ymin, xmax, ymax;

		Voronator(delaunator::DelaunayFromDelaunator * _delau, const double _xmin, const double _ymin, const double _xmax, const double _ymax);

		void cell(std::size_t i, std::vector <double> & _points);
		void clip(std::size_t i, std::vector <double> & _points);
		void clipFinite(std::size_t i, std::vector <double> & _points);
		void clipInfinite(std::size_t i, std::vector <double> & _points, const double vx0, const double vy0, const double vxn, const double vyn);
		std::size_t clipSegment(double x0, double y0, double x1, double y1, unsigned char c0, unsigned char c1, double & sx0, double & sy0, double & sx1, double & sy1);
		std::size_t edge(std::size_t i, unsigned char e0, unsigned char e1, std::vector <double> & P, std::size_t j);
		bool contains(std::size_t i, double x, double y);
		void neighbors(std::uint32_t, std::vector <uint32_t> &, std::uint32_t);

		std::size_t project(double x0, double y0, double vx, double vy, double & x, double & y);
		unsigned char edgecode(double x, double y);
		unsigned char regioncode(double x, double y);

		inline const poca::core::Vec3md * getCellPoints() const { return (poca::core::Vec3md *)cells.data(); }
		inline uint32_t* getFirstPointCells() const { return (unsigned int *)firsts.data(); }
		inline uint32_t* getIndexNeighbors() const { return (unsigned int *)neighs.data(); }

		inline const uint32_t getNSites() const { return (int)(delaunay->points.size() / 2); }
		inline const uint32_t nbEdges() const { return (int)neighs.size(); }

		inline const std::vector <bool>& getBorderCells() const { return borderCells; }
	};
}

