/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayOnSphere.cpp
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

#include <CGAL/squared_distance_3.h>
#include <fstream>

#include "DelaunayOnSphere.hpp"

DelaunayOnSphere::DelaunayOnSphere(const std::vector <poca::core::Vec3mf>& _points, const poca::core::Vec3mf& _center, const float _radius):m_delaunay(NULL), m_centroid(_center), m_radius(_radius)
{
	SphereTraits traits(SpherePoint_3(_center.x(), _center.y(), _center.z()), _radius);
	m_delaunay = new CGALDelaunayOnSphere(traits);
	SphereTraits::Construct_point_on_sphere_2 cst = traits.construct_point_on_sphere_2_object();
	for (const poca::core::Vec3mf& p : _points){
		SpherePoint_3 pt(p.x(), p.y(), p.z());
		m_delaunay->insert(cst(pt));
	}

	std::cout << m_delaunay->number_of_vertices() << " vertices" << std::endl;
	std::cout << m_delaunay->number_of_faces() << " solid faces" << std::endl;
}

DelaunayOnSphere::~DelaunayOnSphere()
{
	if (m_delaunay != NULL)
		delete m_delaunay;
}

void DelaunayOnSphere::generateTriangles(std::vector <poca::core::Vec3mf>& _triangles) const
{
	_triangles.clear();
	for (CGALDelaunayOnSphere::Finite_faces_iterator it = m_delaunay->finite_faces_begin(); it != m_delaunay->finite_faces_end(); it++) {
		for (int i = 0; i < 3; i++) {
			CGALDelaunayOnSphere::Vertex_handle v = it->vertex(i);
			_triangles.push_back(poca::core::Vec3mf(v->point().x(), v->point().y(), v->point().z()));
		}
	}
}

void DelaunayOnSphere::generateVoronoi(std::vector <poca::core::Vec3mf>& _edges, std::vector <poca::core::Vec3mf>& _normals, std::vector <poca::core::Vec3mf>& _triangles) const
{
	_edges.clear();
	_normals.clear();
	_triangles.clear();
	int n = 0;
	for (CGALDelaunayOnSphere::Finite_vertices_iterator it = m_delaunay->finite_vertices_begin(); it != m_delaunay->finite_vertices_end(); it++) {
		poca::core::Vec3mf vertex(it->point().x(), it->point().y(), it->point().z()), vector(vertex - m_centroid);
		vector.normalize();
		CGALDelaunayOnSphere::Edge_circulator first = m_delaunay->incident_edges(it), current = first;
		do {
			SphereSegment_3 dual = m_delaunay->dual(current);
			poca::core::Vec3mf p1(dual.source().x(), dual.source().y(), dual.source().z()), p2(dual.target().x(), dual.target().y(), dual.target().z());
			poca::core::Vec3mf v1(p1 - vertex), v2(p2 - vertex);
			v1.normalize();
			v2.normalize();
			poca::core::Vec3mf cross = v1.cross(v2), normal(vector.normalize());
			if (cross.dot(vector) > 0) {
				_edges.push_back(poca::core::Vec3mf(p1));
				_edges.push_back(poca::core::Vec3mf(p2));
	
				_normals.push_back(normal);
				_normals.push_back(normal);

				_triangles.push_back(poca::core::Vec3mf(p1));
				_triangles.push_back(poca::core::Vec3mf(p2));
				_triangles.push_back(vertex);

			}
			else {
				_edges.push_back(poca::core::Vec3mf(p2));
				_edges.push_back(poca::core::Vec3mf(p1));

				_normals.push_back(normal);
				_normals.push_back(normal);

				_triangles.push_back(poca::core::Vec3mf(p2));
				_triangles.push_back(poca::core::Vec3mf(p1));
				_triangles.push_back(vertex);

			}
			current++;
		} while (current != first);
		n++;
	}
}

void DelaunayOnSphere::generateFeatureVoronoi(const std::vector <float>& _featuresLocs, std::vector <float>& _features) const
{
	_features.clear();
	int n = 0, tmp = m_delaunay->number_of_vertices();
	for (CGALDelaunayOnSphere::Finite_vertices_iterator it = m_delaunay->finite_vertices_begin(); it != m_delaunay->finite_vertices_end(); it++) {
		CGALDelaunayOnSphere::Edge_circulator first = m_delaunay->incident_edges(it), current = first;
		do {
			_features.push_back(_featuresLocs[n]);
			_features.push_back(_featuresLocs[n]);
			_features.push_back(_featuresLocs[n]);
			current++;
		} while (current != first);
		n++;
	}
}