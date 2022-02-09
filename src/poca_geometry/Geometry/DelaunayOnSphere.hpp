/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayOnSphere.hpp
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

#ifndef DelaunayOnSphere_hpp__
#define DelaunayOnSphere_hpp__

#include <vector>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_on_sphere_traits_2.h>
#include <CGAL/Delaunay_triangulation_on_sphere_2.h>
#include <CGAL/Projection_on_sphere_traits_3.h>

#include <General/Vec3.hpp>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
#define WITH_PROJ
#ifdef WITH_PROJ
typedef CGAL::Projection_on_sphere_traits_3<K>  SphereTraits;
#else
typedef CGAL::Delaunay_triangulation_on_sphere_traits_2<K>  SphereTraits;
#endif
typedef CGAL::Delaunay_triangulation_on_sphere_2<SphereTraits>    CGALDelaunayOnSphere;
typedef SphereTraits::Point_3                                     SpherePoint_3;
typedef SphereTraits::Segment_3                                   SphereSegment_3;


class DelaunayOnSphere {
public:
	DelaunayOnSphere(const std::vector <poca::core::Vec3mf>&, const poca::core::Vec3mf&, const float);
	~DelaunayOnSphere();

	void generateTriangles(std::vector <poca::core::Vec3mf>&) const;
	void generateVoronoi(std::vector <poca::core::Vec3mf>&, std::vector <poca::core::Vec3mf>&, std::vector <poca::core::Vec3mf>&) const;
	void generateFeatureVoronoi(const std::vector <float>&, std::vector <float>&) const;

protected:
	CGALDelaunayOnSphere* m_delaunay;
	poca::core::Vec3mf m_centroid;
	float m_radius;
};

#endif