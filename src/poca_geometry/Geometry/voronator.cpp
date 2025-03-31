/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      voronator.cpp
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

#include "voronator.hpp"

namespace voronator {
	Voronator::Voronator(delaunator::DelaunayFromDelaunator* _delau,
		const double _xmin,
		const double _ymin,
		const double _xmax,
		const double _ymax) : delaunay(_delau), delaunator(_delau->delaunator), xmin(_xmin), ymin(_ymin), xmax(_xmax), ymax(_ymax) {
		circumcenters.resize(delaunator->triangles.size() / 3 * 2);
		vectors.resize(_delau->points.size() * 2);

		std::size_t i = 0, j = 0;
		double x, y;
		for (; i < _delau->delaunator->triangles.size(); i += 3, j += 2) {
			const std::size_t t1 = delaunator->triangles[i] * 2;
			const std::size_t t2 = delaunator->triangles[i + 1] * 2;
			const std::size_t t3 = delaunator->triangles[i + 2] * 2;
			const double x1 = _delau->points[t1];
			const double y1 = _delau->points[t1 + 1];
			const double x2 = _delau->points[t2];
			const double y2 = _delau->points[t2 + 1];
			const double x3 = _delau->points[t3];
			const double y3 = _delau->points[t3 + 1];

			const double dx = x2 - x1;
			const double dy = y2 - y1;
			const double ex = x3 - x1;
			const double ey = y3 - y1;
			const double bl = dx * dx + dy * dy;
			const double cl = ex * ex + ey * ey;
			const double ab = (dx * ey - dy * ex) * 2;

			if (ab == 0) {
				// degenerate case (collinear diagram)
				x = (x1 + x3) / 2 - 1e8 * ey;
				y = (y1 + y3) / 2 + 1e8 * ex;
			}
			else if (fabs(ab) < 1e-8) {
				// almost equal points (degenerate triangle)
				x = (x1 + x3) / 2;
				y = (y1 + y3) / 2;
			}
			else {
				const double d = 1 / ab;
				x = x1 + (ey * bl - dy * cl) * d;
				y = y1 + (dx * cl - ex * bl) * d;
			}
			circumcenters[j] = x;
			circumcenters[j + 1] = y;
		}

		// Compute exterior cell rays.
		size_t h = delaunator->hull[delaunator->hull.size() - 1];
		size_t p0, p1 = h * 4;
		double x0, x1 = _delau->points[2 * h];
		double y0, y1 = _delau->points[2 * h + 1];
		std::fill(vectors.begin(), vectors.end(), 0);
		for (std::size_t i = 0; i < delaunator->hull.size(); ++i) {
			h = delaunator->hull[i];
			p0 = p1; x0 = x1; y0 = y1;
			p1 = h * 4; x1 = _delau->points[2 * h]; y1 = _delau->points[2 * h + 1];
			vectors[p0 + 2] = vectors[p1] = y0 - y1;
			vectors[p0 + 3] = vectors[p1 + 1] = x1 - x0;
		}
		std::size_t ncells = delaunay->points.size() / 2, cpt = 0;
		borderCells.resize(ncells);
		std::fill(borderCells.begin(), borderCells.end(), false);
		std::vector <double> cellPoints;
		cells.resize(2 * 20 * ncells);
		firsts.resize(ncells + 1);
		firsts[0] = 0;
		for (std::size_t n = 0; n < ncells; n++) {
			clip(n, cellPoints);
			firsts[n + 1] = cpt + cellPoints.size();
			if (cpt + cellPoints.size() > cells.size())
				cells.resize(2 * cells.size());
			std::copy(cellPoints.begin(), cellPoints.end(), cells.begin() + cpt);
			cpt += cellPoints.size();
		}
		cells.resize(cpt);
		cells.shrink_to_fit();

		cpt = 0;
		std::vector <uint32_t> neighs_cell;
		neighs.resize(cells.size() / 2);
		for (std::size_t n = 0; n < ncells; n++) {
			size_t size = firsts[n + 1] - firsts[n];
			neighbors(n, neighs_cell, size / 2);
			std::copy(neighs_cell.begin(), neighs_cell.end(), neighs.begin() + cpt);
			cpt += neighs_cell.size();
		}

		std::vector <double> cellsTmp(cells);
		cells.clear();
		for (size_t n = 0; n < cellsTmp.size(); n += 2) {
			cells.push_back(cellsTmp[n]);
			cells.push_back(cellsTmp[n + 1]);
			cells.push_back(0.);
		}
	}

	void Voronator::clip(std::size_t i, std::vector <double> & _points) {
		// degenerate case (1 valid point: return the box)
		if (i == 0 && delaunator->hull.size() == 1) {
			double tmp[] = { xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin };
			_points.assign(tmp, tmp + 8);
			return;
		}
		cell(i, _points);
		if (_points.empty()) return;
		std::size_t v = i * 4;
		if (vectors[v] || vectors[v + 1])
			clipInfinite(i, _points, vectors[v], vectors[v + 1], vectors[v + 2], vectors[v + 3]);
		else
			clipFinite(i, _points);
	}

	void Voronator::cell(std::size_t i, std::vector <double> & _points) {
		std::size_t e0 = delaunay->inedges[i];
		if (e0 == -1) return; // coincident point
		_points.clear();
		std::size_t e = e0;
		do {
			std::size_t t = floor(e / 3);
			_points.push_back(circumcenters[t * 2]);
			_points.push_back(circumcenters[t * 2 + 1]);
			e = e % 3 == 2 ? e - 2 : e + 1;
			if (delaunator->triangles[e] != i) break; // bad triangulation
			e = delaunator->halfedges[e];
		} while (e != e0 && e != -1);
	}

	void Voronator::clipFinite(std::size_t i, std::vector <double> & _points) {
		std::size_t n = _points.size();
		std::vector <double> P;
		double x0, y0, x1 = _points[n - 2], y1 = _points[n - 1];
		unsigned char c0, c1 = regioncode(x1, y1);
		unsigned char e0, e1 = 0;
		double sx0, sy0, sx1, sy1;
		bool cut = false;
		for (std::size_t j = 0; j < n; j += 2) {
			x0 = x1; y0 = y1; x1 = _points[j]; y1 = _points[j + 1];
			c0 = c1; c1 = regioncode(x1, y1);
			if (c0 == 0 && c1 == 0) {
				e0 = e1; e1 = 0;
				P.push_back(x1);
				P.push_back(y1);
			}
			else {
				cut = true;
				if (c0 == 0) {
					if (clipSegment(x0, y0, x1, y1, c0, c1, sx0, sy0, sx1, sy1) == 0) continue;
				}
				else {
					if (clipSegment(x1, y1, x0, y0, c1, c0, sx1, sy1, sx0, sy0) == 0) continue;
					e0 = e1; e1 = edgecode(sx0, sy0);
					if (e0 && e1) edge(i, e0, e1, P, P.size());
					P.push_back(sx0);
					P.push_back(sy0);
				}
				e0 = e1; e1 = edgecode(sx1, sy1);
				if (e0 && e1) edge(i, e0, e1, P, P.size());
				P.push_back(sx1);
				P.push_back(sy1);
			}
		}
		if (!P.empty()) {
			e0 = e1; e1 = edgecode(P[0], P[1]);
			if (e0 && e1) edge(i, e0, e1, P, P.size());
		}
		else if (contains(i, (xmin + xmax) / 2, (ymin + ymax) / 2)) {
			double tmp[] = { xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin };
			P.assign(tmp, tmp + 8);
		}
		_points.assign(P.begin(), P.end());
		if (cut)
			borderCells[i] = true;
	}

	void Voronator::clipInfinite(std::size_t i, std::vector <double> & _points, const double vx0, const double vy0, const double vxn, const double vyn) {
		std::vector <double> P(_points);
		double x, y;
		if (project(P[0], P[1], vx0, vy0, x, y) != 0) {
			P.insert(P.begin(), y);
			P.insert(P.begin(), x);
		}
		if (project(P[P.size() - 2], P[P.size() - 1], vxn, vyn, x, y)) {
			P.push_back(x);
			P.push_back(y);
		}
		clipFinite(i, P);
		if (!P.empty()) {
			std::size_t n = P.size();
			unsigned char c0, c1 = edgecode(P[n - 2], P[n - 1]);
			for (std::size_t j = 0; j < n; j += 2) {
				c0 = c1; c1 = edgecode(P[j], P[j + 1]);
				if (c0 && c1) {
					j = edge(i, c0, c1, P, j);
					n = P.size();
				}
			}
		}
		else if (contains(i, (xmin + xmax) / 2, (ymin + ymax) / 2)) {
			double tmp[] = { xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax };
			P.assign(tmp, tmp + 8);
		}
		_points.assign(P.begin(), P.end());

		borderCells[i] = true;
	}

	std::size_t Voronator::clipSegment(double x0, double y0, double x1, double y1, unsigned char c0, unsigned char c1, double & sx0, double & sy0, double & sx1, double & sy1) {
		while (true) {
			if (c0 == 0 && c1 == 0) {
				sx0 = x0; sy0 = y0; sx1 = x1; sy1 = y1;
				return 1;
			}
			if (c0 && c1) return 0;
			double x, y;
			unsigned char c = c0 | c1;
			if (c & 0b1000) {
				x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0);
				y = ymax;
			}
			else if (c & 0b0100) {
				x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0);
				y = ymin;
			}
			else if (c & 0b0010) {
				y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0);
				x = xmax;
			}
			else {
				y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0);
				x = xmin;
			}
			if (c0) {
				x0 = x;
				y0 = y;
				c0 = regioncode(x0, y0);
			}
			else {
				x1 = x;
				y1 = y;
				c1 = regioncode(x1, y1);
			}
		}
	}

	std::size_t Voronator::edge(std::size_t i, unsigned char e0, unsigned char e1, std::vector <double> & P, std::size_t j) {
		while (e0 != e1) {
			double x, y;
			switch (e0) {
			case 0b0101: e0 = 0b0100; continue; // top-left
			case 0b0100: e0 = 0b0110; x = xmax; y = ymin; break; // top
			case 0b0110: e0 = 0b0010; continue; // top-right
			case 0b0010: e0 = 0b1010; x = xmax; y = ymax; break; // right
			case 0b1010: e0 = 0b1000; continue; // bottom-right
			case 0b1000: e0 = 0b1001; x = xmin; y = ymax; break; // bottom
			case 0b1001: e0 = 0b0001; continue; // bottom-left
			case 0b0001: e0 = 0b0101; x = xmin; y = ymin; break; // left
			}
			if (/*(P[j] != x || P[j + 1] != y) &&*/ contains(i, x, y)) {
				P.insert(P.begin() + j, y);
				P.insert(P.begin() + j, x);
				j += 2;
			}
		}
		if (P.size() > 4) {
			for (std::size_t i = 0; i < P.size(); i += 2) {
				std::size_t j = (i + 2) % P.size(), k = (i + 4) % P.size();
				if (P[i] == P[j] && P[j] == P[k] || P[i + 1] == P[j + 1] && P[j + 1] == P[k + 1]) {
					P.erase(P.begin() + j, P.begin() + j + 2);
					i -= 2;
				}
			}
		}
		return j;
	}

	bool Voronator::contains(std::size_t i, double x, double y) {
		if ((x = +x, x != x) || (y = +y, y != y)) return false;
		return delaunay->step(i, x, y) == i;
	}

	//NbToAchieve is added for compatibilities with Tesseler soft. It should not be used.
	//We need it because the number of neighs per cell has to be the same number of edges of its polygon
	void Voronator::neighbors(std::uint32_t i, std::vector <uint32_t> & neighs, std::uint32_t nbToAchieve) {
		neighs.clear();
		uint32_t sizeI = firsts[i + 1] - firsts[i];
		if (sizeI == 0) return;

		double * ci = cells.data() + firsts[i];
		std::vector <std::uint32_t> delau_neighs;
		delaunay->neighborsUint32(i, delau_neighs);

		for (std::size_t jt = 0; jt < delau_neighs.size(); jt++) {
			std::uint32_t j = delau_neighs[jt];
			uint32_t sizeJ = firsts[j + 1] - firsts[j];
			if (sizeJ == 0) continue;
			double* cj = cells.data() + firsts[j];

			bool done = false;
			std::uint32_t li = sizeI, lj = sizeJ;
			for (std::uint32_t ai = 0; ai < li && !done; ai += 2) {
				for (std::uint32_t aj = 0; aj < lj && !done; aj += 2) {
					if (ci[ai] == cj[aj]
						&& ci[ai + 1] == cj[aj + 1]
						&& ci[(ai + 2) % li] == cj[(aj + lj - 2) % lj]
						&& ci[(ai + 3) % li] == cj[(aj + lj - 1) % lj]
						) {
						neighs.push_back(j);
						done = true;
					}
				}
			}
		}
		for(std::uint32_t n = neighs.size(); n < nbToAchieve; n++)
			neighs.push_back(std::numeric_limits < uint32_t >::max());
	}

	std::size_t Voronator::project(double x0, double y0, double vx, double vy, double & x, double & y) {
		double t = DBL_MAX, c;
		if (vy < 0) { // top
			if (y0 <= ymin) return 0;
			if ((c = (ymin - y0) / vy) < t) {
				y = ymin;
				x = x0 + (t = c) * vx;
			}
		}
		else if (vy > 0) { // bottom
			if (y0 >= ymax) return 0;
			if ((c = (ymax - y0) / vy) < t) {
				y = ymax;
				x = x0 + (t = c) * vx;
			}
		}
		if (vx > 0) { // right
			if (x0 >= xmax) return 0;
			if ((c = (xmax - x0) / vx) < t) {
				x = xmax;
				y = y0 + (t = c) * vy;
			}
		}
		else if (vx < 0) { // left
			if (x0 <= xmin) return 0;
			if ((c = (xmin - x0) / vx) < t) {
				x = xmin;
				y = y0 + (t = c) * vy;
			}
		}
		return 1;
	}

	unsigned char Voronator::edgecode(double x, double y) {
		return (x == xmin ? 0b0001 : x == xmax ? 0b0010 : 0b0000) | (y == ymin ? 0b0100 : y == ymax ? 0b1000 : 0b0000);
	}
	unsigned char Voronator::regioncode(double x, double y) {
		unsigned char tx = (x < xmin ? 0b0001 : x > xmax ? 0b0010 : 0b0000);
		unsigned char ty = (y < ymin ? 0b0100 : y > ymax ? 0b1000 : 0b0000);
		unsigned char tres = tx | ty;
		return (x < xmin ? 0b0001 : x > xmax ? 0b0010 : 0b0000) | (y < ymin ? 0b0100 : y > ymax ? 0b1000 : 0b0000);
	}
}

