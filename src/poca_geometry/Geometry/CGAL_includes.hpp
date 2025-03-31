/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CGAL_includes.hpp
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

#ifndef CGAL_includes_h__
#define CGAL_includes_h__

#include <CGAL/version.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Cartesian.h>
#include <CGAL/Min_ellipse_2.h>
#include <CGAL/Min_ellipse_2_traits_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/number_utils.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_with_circumcenter_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_on_sphere_traits_2.h>
#include <CGAL/Delaunay_triangulation_on_sphere_2.h>
#include <CGAL/Projection_on_sphere_traits_3.h>

struct FaceInfo2
{
	FaceInfo2() {}
	int m_nesting_level;
	bool in_domain() {
		return m_nesting_level % 2 == 1;
	}
};

typedef CGAL::Exact_predicates_inexact_constructions_kernel K_inexact;

typedef K_inexact::Point_2 Point_2;
typedef K_inexact::Segment_2 Segment_2;
typedef K_inexact::Ray_2 Ray_2;
typedef K_inexact::Line_2 Line_2;
typedef K_inexact::Triangle_2 Triangle_2;
typedef K_inexact::Iso_rectangle_2 Iso_rectangle_2;
typedef CGAL::Triangulation_vertex_base_with_info_2 < int, K_inexact > Vb_int;
typedef CGAL::Triangulation_vertex_base_with_info_2 < bool, K_inexact > Vb_bool;
typedef CGAL::Triangulation_face_base_with_info_2 < int, K_inexact > Fb;
typedef CGAL::Triangulation_data_structure_2 < Vb_int, Fb > Tds_int;
typedef CGAL::Triangulation_data_structure_2 < Vb_bool, Fb > Tds_bool;
typedef CGAL::Delaunay_triangulation_2<K_inexact, Tds_int> Delaunay_triangulation_2;
typedef CGAL::Delaunay_triangulation_2<K_inexact, Tds_bool> Delaunay_triangulation_2_bool;
//For constrained triangulation
typedef CGAL::Triangulation_face_base_with_info_2 < FaceInfo2, K_inexact > Fbb;
typedef CGAL::Constrained_triangulation_face_base_2 < K_inexact, Fbb > ConstFb;
typedef CGAL::Triangulation_data_structure_2 < Vb_int, ConstFb > ConstTds;
typedef CGAL::Exact_predicates_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2 < K_inexact, ConstTds, Itag > Constrained_Delaunay_triangulation_2;
typedef Constrained_Delaunay_triangulation_2::Point ConstPoint;
typedef CGAL::Polygon_2 < K_inexact > Polygon_2;
typedef std::list<Polygon_2> Polygons_2;
typedef Delaunay_triangulation_2::Face_handle FaceHandle;
typedef Delaunay_triangulation_2::Vertex_handle VertHandle;
typedef Delaunay_triangulation_2::Vertex Vert2;
typedef Delaunay_triangulation_2::Edge_circulator EdgeCirc;
typedef Delaunay_triangulation_2::Edge Edge;
typedef Delaunay_triangulation_2::Finite_edges_iterator Finite_edges_iterator;
typedef Delaunay_triangulation_2_bool::Face_handle FaceHandle_bool;
typedef Delaunay_triangulation_2_bool::Vertex_handle VertHandle_bool;
typedef Delaunay_triangulation_2_bool::Vertex Vert2_bool;
typedef Delaunay_triangulation_2_bool::Edge_circulator EdgeCirc_bool;
typedef Delaunay_triangulation_2_bool::Edge Edge_bool;
typedef Delaunay_triangulation_2_bool::Finite_edges_iterator Finite_edges_iterator_bool;

typedef K_inexact::Point_3 Point_3_inexact;
typedef K_inexact::Segment_3 Segment_3_inexact;
typedef K_inexact::Ray_3 Ray_3_inexact;
typedef K_inexact::Plane_3 Plane_3_inexact;
typedef K_inexact::Vector_3 Vector_3_inexact;
typedef std::pair<Point_3_inexact, Vector_3_inexact> pointWnormal_3_inexact;
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(5,0,0)
typedef CGAL::Triangulation_vertex_base_with_info_3 < int, K_inexact > Vb_int_3_inexact;
typedef CGAL::Delaunay_triangulation_cell_base_3<K_inexact> Cb_3_inexact;
typedef CGAL::Delaunay_triangulation_cell_base_with_circumcenter_3<K_inexact, Cb_3_inexact>  Cbc_3_inexact; // Now, the cell provides both info() and circumcenter()
typedef CGAL::Triangulation_cell_base_with_info_3 < int, K_inexact, Cbc_3_inexact > Cbc_int_3_inexact;
typedef CGAL::Triangulation_data_structure_3<Vb_int_3_inexact, Cbc_int_3_inexact, CGAL::Parallel_tag> Tds_3_inexact;
typedef CGAL::Delaunay_triangulation_3<K_inexact, Tds_3_inexact> Triangulation_3_inexact;
#else
typedef CGAL::Triangulation_vertex_base_with_info_3 < int, K_inexact > Vb_int_3_inexact;
typedef CGAL::Triangulation_cell_base_with_info_3 < int, K_inexact > Cb_3_inexact;
typedef CGAL::Triangulation_data_structure_3<Vb_int_3_inexact, Cb_3_inexact, CGAL::Parallel_tag> Tds_3_inexact;
typedef CGAL::Delaunay_triangulation_3<K_inexact, Tds_3_inexact> Triangulation_3_inexact;
#endif
typedef Triangulation_3_inexact::Point Point_delau_3D_inexact;
typedef Triangulation_3_inexact::Finite_vertices_iterator Finite_vertices_iterator_3_inexact;
typedef Triangulation_3_inexact::Finite_cells_iterator Finite_cells_iterator_3_inexact;
typedef Triangulation_3_inexact::Finite_facets_iterator Finite_facets_iterator_3_inexact;
typedef Triangulation_3_inexact::Cell_iterator Cell_iterator_3_inexact;
typedef Triangulation_3_inexact::Cell_handle Cell_handle_3_inexact;
typedef Triangulation_3_inexact::Vertex_handle Vertex_handle_3_inexact;
typedef Triangulation_3_inexact::Facet_circulator Facet_circulator_3_inexact;
typedef Triangulation_3_inexact::Cell_circulator Cell_circulator_3_inexact;
typedef Triangulation_3_inexact::Vertex_iterator Vertex_iterator_3_inexact;
typedef Triangulation_3_inexact::Edge Edge_3_inexact;
typedef Triangulation_3_inexact::Edge_iterator edge_iterator_3_inexact;
typedef Triangulation_3_inexact::Facet Facet_3_inexact;

typedef CGAL::Simple_cartesian< double >  Kernel;
typedef Kernel::Line_2 KernelLine;
typedef Kernel::Point_2 KernelPoint;
typedef Kernel::Line_3 KernelLine3D;
typedef Kernel::Point_3 KernelPoint3D;
typedef Kernel::Plane_3 KernelPlane3D;
typedef Kernel::Vector_3 KernelVector3D;

typedef CGAL::Exact_predicates_exact_constructions_kernel K_exact;
typedef K_exact::Point_2 Point_22;
typedef CGAL::Polygon_2<K_exact> Polygon_22;
typedef CGAL::Polygon_with_holes_2<K_exact> Polygon_with_holes_2;
typedef std::list<Polygon_with_holes_2> Pwh_list_2;

typedef K_exact::Point_3 Point_3_exact;
typedef K_exact::Segment_3 Segment_3_exact;
typedef K_exact::Ray_3 Ray_3_exact;
typedef K_exact::Plane_3 Plane_3_exact;
typedef CGAL::Triangulation_vertex_base_with_info_3 < int, K_exact > Vb_int_3_exact;
typedef CGAL::Triangulation_cell_base_with_info_3 < int, K_exact > Cb_3_exact;
typedef CGAL::Triangulation_data_structure_3<Vb_int_3_exact, Cb_3_exact, CGAL::Parallel_tag> Tds_3_exact;
typedef CGAL::Delaunay_triangulation_3<K_exact, Tds_3_exact> Triangulation_3_exact;
typedef Triangulation_3_exact::Point Point_delau_3D_exact;
typedef Triangulation_3_exact::Finite_vertices_iterator Finite_vertices_iterator_3_exact;
typedef Triangulation_3_exact::Finite_cells_iterator Finite_cells_iterator_3_exact;
typedef Triangulation_3_exact::Finite_facets_iterator Finite_facets_iterator_3_exact;
typedef Triangulation_3_exact::Cell_iterator Cell_iterator_3_exact;
typedef Triangulation_3_exact::Cell_handle Cell_handle_3_exact;
typedef Triangulation_3_exact::Vertex_handle Vertex_handle_3_exact;
typedef Triangulation_3_exact::Facet_circulator Facet_circulator_3_exact;
typedef Triangulation_3_exact::Cell_circulator Cell_circulator_3_exact;
typedef Triangulation_3_exact::Vertex_iterator Vertex_iterator_3_exact;
typedef Triangulation_3_exact::Edge Edge_3_exact;
typedef Triangulation_3_exact::Edge_iterator edge_iterator_3_exact;
typedef Triangulation_3_exact::Facet Facet_3_exact;
typedef CGAL::Polyhedron_3<K_exact> Polyhedron_3_exact;
typedef CGAL::Surface_mesh<Point_3_exact> Surface_mesh_3_exact;
typedef CGAL::Nef_polyhedron_3<K_exact> Nef_Polyhedron_3_exact;
typedef CGAL::Polyhedron_3<K_inexact> Polyhedron_3_inexact;
typedef CGAL::Surface_mesh<Point_3_inexact> Surface_mesh_3_inexact;
typedef CGAL::Nef_polyhedron_3<K_inexact> Nef_Polyhedron_3_inexact;


#define WITH_PROJ
#ifdef WITH_PROJ
typedef CGAL::Projection_on_sphere_traits_3<K_inexact>  SphereTraits;
#else
typedef CGAL::Delaunay_triangulation_on_sphere_traits_2<K>  SphereTraits;
#endif
typedef CGAL::Delaunay_triangulation_on_sphere_2<SphereTraits>    CGALDelaunayOnSphere;
typedef SphereTraits::Point_3                                     SpherePoint_3;
typedef SphereTraits::Segment_3                                   SphereSegment_3;

typedef Kernel::Point_3 Point_3_double;
typedef CGAL::Surface_mesh<Point_3_double> Surface_mesh_3_double;
typedef boost::graph_traits<Surface_mesh_3_double>::vertex_descriptor    vertex_descriptor;
typedef boost::graph_traits<Surface_mesh_3_double>::halfedge_descriptor         halfedge_descriptor;
typedef boost::graph_traits<Surface_mesh_3_double>::face_descriptor             face_descriptor;
typedef Surface_mesh_3_double::Property_map<face_descriptor, double> Facet_double_map;
typedef Surface_mesh_3_double::Property_map<face_descriptor, std::size_t> Facet_size_t_map;


#endif // CGAL_includes_h__

