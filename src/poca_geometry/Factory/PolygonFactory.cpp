/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PolygonFactory.cpp
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

#include <Windows.h>
#include <GL/glew.h>
#include <omp.h>
#include <unordered_set>
#include <algorithm>
#include <fstream>

#include <General/MyArray.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <General/PluginList.hpp>

#include "PolygonFactory.hpp"
#include "../Geometry/CGAL_includes.hpp"
#include "../Geometry/StraightLine.hpp"
#include "../Geometry/Polygon.hpp"

namespace poca::geometry {
	void mark_domains(Constrained_Delaunay_triangulation_2* _ct, Constrained_Delaunay_triangulation_2::Face_handle _start, int _index, std::list < Constrained_Delaunay_triangulation_2::Edge >& _border)
	{
		if (_start->info().m_nesting_level != -1)
			return;
		std::list < Constrained_Delaunay_triangulation_2::Face_handle > queue;
		queue.push_back(_start);
		while (!queue.empty()) {
			Constrained_Delaunay_triangulation_2::Face_handle fh = queue.front();
			queue.pop_front();
			if (fh->info().m_nesting_level == -1) {
				fh->info().m_nesting_level = _index;
				for (int i = 0; i < 3; i++) {
					Constrained_Delaunay_triangulation_2::Edge e(fh, i);
					Constrained_Delaunay_triangulation_2::Face_handle n = fh->neighbor(i);
					if (n->info().m_nesting_level == -1) {
						if (_ct->is_constrained(e)) _border.push_back(e);
						else queue.push_back(n);
					}
				}
			}
		}
	}

	void mark_domains(Constrained_Delaunay_triangulation_2* _cdt)
	{
		for (Constrained_Delaunay_triangulation_2::All_faces_iterator it = _cdt->all_faces_begin(); it != _cdt->all_faces_end(); ++it)
			it->info().m_nesting_level = -1;
		std::list < Constrained_Delaunay_triangulation_2::Edge > border;
		mark_domains(_cdt, _cdt->infinite_face(), 0, border);
		while (!border.empty()) {
			Constrained_Delaunay_triangulation_2::Edge e = border.front();
			border.pop_front();
			Constrained_Delaunay_triangulation_2::Face_handle n = e.first->neighbor(e.second);
			if (n->info().m_nesting_level == -1) {
				mark_domains(_cdt, n, e.first->info().m_nesting_level + 1, border);
			}
		}
	}

	PolygonFactoryInterface* createPolygonFactory() {
		return new PolygonFactory();
	}

	PolygonFactory::PolygonFactory()
	{

	}

	PolygonFactory::~PolygonFactory()
	{

	}

	PolygonInterface* PolygonFactory::createPolygon(const std::vector <poca::core::Vec2mf>& _edges)
	{
		//Creation of the polygon representing the outline of the object
		std::vector < poca::core::Vec2mf > pts;
		Constrained_Delaunay_triangulation_2* delau = new Constrained_Delaunay_triangulation_2;
		for (unsigned int i = 0; i < _edges.size(); i += 2)
			delau->insert_constraint(Constrained_Delaunay_triangulation_2::Point(_edges[i].x(), _edges[i].y()), Constrained_Delaunay_triangulation_2::Point(_edges[i + 1].x(), _edges[i + 1].y()));
		mark_domains(delau);

		Constrained_Delaunay_triangulation_2::Vertex_handle first = delau->incident_vertices(delau->infinite_vertex()), current = first, next = NULL, prec = NULL;
#ifndef NDEBUG
		std::cout << "First loc -> ";
		std::cout << first->point().x() << ", " << first->point().y() << std::endl;
#endif
		bool found = false;
		do {
			found = false;
			Constrained_Delaunay_triangulation_2::Edge_circulator ecirc_first = delau->incident_edges(current);
			//try to find the edge that connect current vertex to prec, intuition: if we find this edge we should be able to avoid
			//the problem of having holes in an object with a loc connecting two triangles
			Constrained_Delaunay_triangulation_2::Vertex_handle tmp1, tmp2;
			tmp1 = ecirc_first->first->vertex((ecirc_first->second + 1) % 3);
			tmp2 = ecirc_first->first->vertex((ecirc_first->second + 2) % 3);
#ifndef NDEBUG
			std::cout << "First edge -> [" << tmp1->point().x() << ", " << tmp1->point().y() << "], [" << tmp2->point().x() << ", " << tmp2->point().y() << "]" << std::endl;
#endif
			Constrained_Delaunay_triangulation_2::Edge_circulator ecirc_cur = ecirc_first;
			if (prec != NULL) {
				while (!((tmp1 == current && tmp2 == prec) || (tmp1 == prec && tmp2 == current))) {
					ecirc_first++;
					tmp1 = ecirc_first->first->vertex((ecirc_first->second + 1) % 3);
					tmp2 = ecirc_first->first->vertex((ecirc_first->second + 2) % 3);
#ifndef NDEBUG
					std::cout << "Changing First edge -> [" << tmp1->point().x() << ", " << tmp1->point().y() << "], [" << tmp2->point().x() << ", " << tmp2->point().y() << "]" << std::endl;
#endif
				}
			}
			else {
				//If prec is NULL then we are trying to determine the first point of the outline. We need to find a point connected to only two edges (if the point is connected to a hole, there will be more than two edges
				//then we get the two edges and try to determine the good orientation (ccw) in order to go in the right direction
				Constrained_Delaunay_triangulation_2::Edge_circulator edges[2];
				unsigned int nbEdges = 0;
				bool correctEdgeFound = false;
				while (!correctEdgeFound) {
					do {
						Constrained_Delaunay_triangulation_2::Face_handle f1 = ecirc_cur->first, f2 = ecirc_cur->first->neighbor(ecirc_cur->second);
						bool inDomain1 = !delau->is_infinite(f1), inDomain2 = !delau->is_infinite(f2);
						if (inDomain1) inDomain1 = f1->info().in_domain();
						if (inDomain2) inDomain2 = f2->info().in_domain();
						if (inDomain1 != inDomain2)
							if (nbEdges < 2)
								edges[nbEdges++] = ecirc_cur;
						ecirc_cur++;
					} while (ecirc_cur != ecirc_first && !correctEdgeFound);
					if (nbEdges == 2) {
						correctEdgeFound = true;
						poca::core::Vec2mf edgesVector[2];
						for (unsigned i = 0; i < 2; i++) {
							tmp1 = edges[i]->first->vertex((edges[i]->second + 1) % 3);
							tmp2 = edges[i]->first->vertex((edges[i]->second + 2) % 3);
							if (tmp1 == current)
								edgesVector[i].set(tmp2->point().x() - current->point().x(), tmp2->point().y() - current->point().y());
							else if (tmp2 == current)
								edgesVector[i].set(tmp1->point().x() - current->point().x(), tmp1->point().y() - current->point().y());
							edgesVector[i].normalize();
						}
#ifndef NDEBUG
						std::cout << "Vector 1 -> [" << edgesVector[0].x() << ", " << edgesVector[0].y() << "]" << std::endl;
						std::cout << "Vector 2 -> [" << edgesVector[1].x() << ", " << edgesVector[1].y() << "]" << std::endl;
#endif
						double angle = edgesVector[0].findSignedAngle(edgesVector[1]);
						double crossP = edgesVector[0].x() * edgesVector[1].y() - edgesVector[0].y() * edgesVector[1].x();
						if (crossP < 0)
							ecirc_first = edges[1];
						else
							ecirc_first = edges[0];
						tmp1 = edges[0]->first->vertex((edges[0]->second + 2) % 3);
						tmp2 = edges[0]->first->vertex((edges[0]->second + 1) % 3);
#ifndef NDEBUG
						std::cout << "Edge 1 -> [" << tmp1->point().x() << ", " << tmp1->point().y() << "], [" << tmp2->point().x() << ", " << tmp2->point().y() << "]" << std::endl;
#endif
						tmp1 = edges[1]->first->vertex((edges[1]->second + 2) % 3);
						tmp2 = edges[1]->first->vertex((edges[1]->second + 1) % 3);
#ifndef NDEBUG
						std::cout << "Edge 2 -> [" << tmp1->point().x() << ", " << tmp1->point().y() << "], [" << tmp2->point().x() << ", " << tmp2->point().y() << "]" << std::endl;
						std::cout << "Value angle -> " << angle << ", cross = " << crossP << std::endl;
#endif
						tmp1 = ecirc_first->first->vertex((ecirc_first->second + 1) % 3);
						tmp2 = ecirc_first->first->vertex((ecirc_first->second + 2) % 3);
#ifndef NDEBUG
						std::cout << "2. Changing First edge -> [" << tmp1->point().x() << ", " << tmp1->point().y() << "], [" << tmp2->point().x() << ", " << tmp2->point().y() << "]" << std::endl;
#endif
					}
					else {
						first++;
						current = first;
						ecirc_first = delau->incident_edges(current);
						ecirc_cur = ecirc_first;
					}
				}
			}
			ecirc_cur = ecirc_first;
			do {
				Constrained_Delaunay_triangulation_2::Face_handle f1 = ecirc_cur->first, f2 = ecirc_cur->first->neighbor(ecirc_cur->second);
				bool inDomain1 = !delau->is_infinite(f1), inDomain2 = !delau->is_infinite(f2);
				if (inDomain1) inDomain1 = f1->info().in_domain();
				if (inDomain2) inDomain2 = f2->info().in_domain();
				if (inDomain1 != inDomain2) {
					next = ecirc_cur->first->vertex(ecirc_cur->first->ccw(ecirc_cur->second));
					if (next != prec) {
						pts.push_back(poca::core::Vec2mf(current->point().x(), current->point().y()));
						found = true;
						tmp1 = ecirc_cur->first->vertex((ecirc_cur->second + 1) % 3);
						tmp2 = ecirc_cur->first->vertex((ecirc_cur->second + 2) % 3);
#ifndef NDEBUG
						std::cout << "Found next edge -> [" << tmp1->point().x() << ", " << tmp1->point().y() << "], [" << tmp2->point().x() << ", " << tmp2->point().y() << "]" << std::endl;
#endif
					}
				}
				ecirc_cur++;
			} while (ecirc_cur != ecirc_first && !found);
			if (found) {
				prec = current;
				current = next;
			}
			else
				std::cout << "Problem in creating outline" << std::endl;
		} while (current != first && found);

		std::vector < poca::core::Vec2mf > ptsTmp;
		for (unsigned int i = 1; i < pts.size(); i++) {
			unsigned int prec = i - 1, current = i % pts.size(), next = (i + 1) % pts.size();
			StraightLine line(pts[prec].x(), pts[prec].y(), pts[next].x(), pts[next].y(), StraightLine::PERPENDICULAR_LINE);
			poca::core::Vec2mf res = line.findPoint(pts[current].x(), pts[current].y(), -0.001);
			ptsTmp.push_back(res);
		}

		PolygonInterface* poly = new Polygon(ptsTmp);

		delete delau;
		return poly;
	}

	void PolygonFactory::computeIntersection(PolygonInterface* _poly1, PolygonInterface* _poly2, PolygonInterfaceList& _intersections)
	{
		if (!_intersections.empty()) {
			for (PolygonInterface* poly : _intersections)
				delete poly;
			_intersections.clear();
		}

		Polygon* poly1Tmp = static_cast <Polygon*>(_poly1);
		if (poly1Tmp == NULL) {
			std::cout << "Computing intersection has failed: polygon 1 is not a proper polygon" << std::endl;
			return;
		}
		Polygon* poly2Tmp = static_cast <Polygon*>(_poly2);
		if (poly2Tmp == NULL) {
			std::cout << "Computing intersection has failed: polygon 2 is not a proper polygon" << std::endl;
			return;
		}

		const Polygon_2& poly1 = poly1Tmp->getCGALPolygon();
		const Polygon_2& poly2 = poly2Tmp->getCGALPolygon();

		CGAL::Cartesian_converter< K_inexact, K_exact > toExact;
		CGAL::Cartesian_converter< K_exact, K_inexact > toInexact;

		//It seems that the polygons needs to be counterclockwise in order that the intersection function works
		//std::cout << "is cluster 1 clockwise ? " << _obj1.is_clockwise_oriented() << std::endl;
		//std::cout << "is cluster 2 clockwise ? " << _obj2.is_clockwise_oriented() << std::endl;

		//Conversion from inexact construction polygon to exact construction polygons
		Polygon_22 exactOutline1, exactOutline2;
		for (Polygon_2::Vertex_const_iterator it = poly1.vertices_begin(); it != poly1.vertices_end(); it++)
			exactOutline1.push_back(toExact(*it));
		for (Polygon_2::Vertex_const_iterator it = poly2.vertices_begin(); it != poly2.vertices_end(); it++)
			exactOutline2.push_back(toExact(*it));

		//Boolean intersection only works on exact construction kernel
		Pwh_list_2 intersections;
		CGAL::intersection(exactOutline1, exactOutline2, std::back_inserter(intersections));

		if (intersections.empty()) return;

		//Conversion from exact construction polygons with holes to inexact construction polygons
		for (Pwh_list_2::const_iterator it = intersections.begin(); it != intersections.end(); it++) {
			Polygon_2 tmp;
			for (Polygon_22::Vertex_const_iterator it2 = it->outer_boundary().vertices_begin(); it2 != it->outer_boundary().vertices_end(); it2++)
				tmp.push_back(toInexact(*it2));
			_intersections.push_back(new Polygon(tmp));
		}

		//Computation of the area of all the intersection inexact construction polygons
		/*double areaInexactIntersections = 0.;
		for (Polygons_2::const_iterator it = inters.begin(); it != inters.end(); it++) {
			//std::cout << "is intersection clockwise ? " << it->is_clockwise_oriented() << std::endl;
			areaInexactIntersections += abs(it->area());
		}

		std::cout << "Area = " << abs(areaInexactIntersections) << ", # intersections = " << inters.size() << std::endl;
		*/
	}

	PolygonInterface* PolygonFactory::createPolygon(const std::vector <poca::core::Vec3mf>& _edges)
	{
		return NULL;
	}
}

