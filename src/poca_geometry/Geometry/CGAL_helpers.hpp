/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CGAL_helpers.hpp
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

#ifndef CGAL_helpers_h__
#define CGAL_helpers_h__

#include <CGAL/Side_of_triangle_mesh.h>

#include "CGAL_includes.hpp"

namespace poca::geometry{

	template <class K>
	inline double to_d(const typename K::FT& v) { return CGAL::to_double(v); }

	template <class K>
	inline double edge_len_cgal(const CGAL::Point_2<K>& p, const CGAL::Point_2<K>& q)
	{
		const double dx = to_d<K>(q.x() - p.x());
		const double dy = to_d<K>(q.y() - p.y());
		return std::sqrt(dx * dx + dy * dy);
	}

	template <class K>
	inline double signed_area2(const CGAL::Point_2<K>& a,
		const CGAL::Point_2<K>& b,
		const CGAL::Point_2<K>& c)
	{
		const double ax = to_d<K>(a.x()), ay = to_d<K>(a.y());
		const double bx = to_d<K>(b.x()), by = to_d<K>(b.y());
		const double cx = to_d<K>(c.x()), cy = to_d<K>(c.y());
		// 2 * signed area
		return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
	}

	// Returns pair: {signed_kappa, abs_kappa}
	template <class K>
	inline std::pair<float, float>
		vertex_curvature_control_polygon(const CGAL::Point_2<K>& A,
			const CGAL::Point_2<K>& B,
			const CGAL::Point_2<K>& C)
	{
		const double ab = edge_len_cgal<K>(A, B);
		const double bc = edge_len_cgal<K>(B, C);
		const double ca = edge_len_cgal<K>(C, A);

		const double s2 = signed_area2<K>(A, B, C); // 2 * signed area
		const double denom = ab * bc * ca;
		const double eps = 1e-14;

		if (denom <= eps) return { 0.0, 0.0 };  // degenerate or nearly collinear

		const double k_abs = (2.0 * std::abs(s2)) / denom; // equals 1/R
		const double k_signed = (s2 >= 0.0) ? k_abs : -k_abs;
		return { (float)k_signed, (float)k_abs };
	}

	template <class T>
	inline double edge_len(const T& p, const T& q)
	{
		const double dx = q.x() - p.x();
		const double dy = q.y() - p.y();
		return std::sqrt(dx * dx + dy * dy);
	}

	template <class T>
	std::pair <float, float> curvature_point(const T& A, const T& B, const T& C)
	{
		const double ab = edge_len(A, B);
		const double bc = edge_len(B, C);
		const double ca = edge_len(C, A);

		const double s2 = (B.x() - A.x()) * (C.y() - A.y()) - (B.y() - A.y()) * (C.x() - A.x());// 2 * signed area

		const double denom = ab * bc * ca;
		const double eps = 1e-14;

		if (denom <= eps) return { 0.0, 0.0 };  // degenerate or nearly collinear

		const double k_abs = (2.0 * std::abs(s2)) / denom; // equals 1/R
		const double k_signed = (s2 >= 0.0) ? k_abs : -k_abs;
		return { (float)k_signed, (float)k_abs };
	}

	// Main function: curvature for each vertex of a closed polygon
	// If first point equals last point, the last is ignored.
	template <class K>
	inline std::vector<std::pair<float, float>>
		control_polygon_curvatures(const CGAL::Polygon_2<K>& poly)
	{
		std::vector<std::pair<float, float>> out;
		const std::size_t N0 = poly.size();
		if (N0 < 3) return out;

		// Handle possible duplicated closing vertex
		std::size_t N = N0;
		if (N0 >= 2 && poly[0] == poly[N0 - 1]) N = N0 - 1;
		if (N < 3) return out;

		out.reserve(N);
		for (std::size_t i = 0; i < N; ++i) {
			const auto& A = poly[(i + N - 1) % N];
			const auto& B = poly[i];
			const auto& C = poly[(i + 1) % N];
			out.push_back(vertex_curvature_control_polygon<K>(A, B, C));
		}
		return out;
	}

	void laplacian_smooth(Surface_mesh_3_double&, int, double = 0.5);

	bool insideMesh(const Surface_mesh_3_double&, float, float, float);
	bool insideMesh(const CGAL::Side_of_triangle_mesh<Surface_mesh_3_double, Kernel>&, float, float, float);

	void meshesIntersectingMesh(const Surface_mesh_3_double&, const std::vector <Surface_mesh_3_double>&, std::vector <bool>&);
	void meshesInsideMeshWithCutting(const Surface_mesh_3_double&, std::vector <Surface_mesh_3_double>&, std::vector <Surface_mesh_3_double>&);
}

#endif