/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListPolygon.cpp
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

#include <algorithm>
#include <execution>
#include <CGAL/centroid.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Triangulation_conformer_2.h>
#include <tinysplinecxx.h>
#include <boost/range/combine.hpp>

#include <General/MyData.hpp>
#include <General/BasicComponent.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/ROIInterface.hpp>
#include <General/Misc.h>
#include <Geometry/CGAL_helpers.hpp>

#include "ObjectListPolygon.hpp"
#include "BasicComputation.hpp"
#include "DelaunayTriangulation.hpp"
#include "../Interfaces/ObjectFeaturesFactoryInterface.hpp"



// Works for dim = 2 or 3.
// For 2D:   kappa = abs(x' * y'' - y' * x'') / ( (x'^2 + y'^2)^(3/2) )
// For 3D:   kappa = ||r' cross r''|| / ||r'||^3
namespace splineutils {

	/*using real_t = tinyspline::real;

	// Euclidean norm of a vector
	inline double norm(const std::vector<real_t>& v)
	{
		double s = 0.0;
		for (double x : v) s += x * x;
		return std::sqrt(s);
	}

	// Cross product for 3D vectors stored in std::vector
	inline std::vector<double> cross3(const std::vector<real_t>& a,
		const std::vector<real_t>& b)
	{
		return {
			static_cast<double>(a[1] * b[2] - a[2] * b[1]),
			static_cast<double>(a[2] * b[0] - a[0] * b[2]),
			static_cast<double>(a[0] * b[1] - a[1] * b[0])
		};
	}

	// Compute Greville abscissae for a clamped or open B-spline.
	// For control point i (0..n-1), xi = (t_{i+1} + ... + t_{i+p}) / p, where p = degree.
	inline std::vector<double> greville_abscissae(const tinyspline::BSpline& s)
	{
		const int p = static_cast<int>(s.degree());                 // degree
		const int nCtrl = static_cast<int>(s.numControlPoints());   // number of control points

		if (p <= 0) throw std::runtime_error("Degree must be >= 1 to compute Greville points.");

		// Knot vector length should be nCtrl + p + 1
		const std::vector<real_t> knots = s.knots();
		if (static_cast<int>(knots.size()) != nCtrl + p + 1)
			throw std::runtime_error("Unexpected knot vector size.");

		std::vector<double> xi;
		xi.reserve(nCtrl);

		for (int i = 0; i < nCtrl; ++i) {
			double sum = 0.0;
			for (int j = 1; j <= p; ++j) {
				sum += static_cast<double>(knots[i + j]);
			}
			xi.push_back(sum / p);
		}

		// Clamp into the curve domain just in case
		tinyspline::Domain dom = s.domain(); // provides min() and max()
		const double umin = static_cast<double>(dom.min());
		const double umax = static_cast<double>(dom.max());
		for (double& u : xi) {
			if (u < umin) u = umin;
			if (u > umax) u = umax;
		}
		return xi;
	}

	// Curvature kappa(u) using TinySpline derivatives
	inline float curvature_at(const tinyspline::BSpline& s, const tinyspline::BSpline& d1, const tinyspline::BSpline& d2, double u)
	{
		const size_t dim = s.dimension();
		if (dim < 2 || dim > 3)
			throw std::runtime_error("Curvature only implemented for 2D or 3D splines.");

		tinyspline::DeBoorNet deriv1 = d1.eval(u);
		tinyspline::DeBoorNet deriv2 = d2.eval(u);

		const auto& d1r = deriv1.result();
		const auto& d2r = deriv2.result();

		double dx = d1r[0], dy = d1r[1];
		double ddx = d2r[0], ddy = d2r[1];

		double num = std::abs(dx * ddy - dy * ddx);
		double denom = std::pow(dx * dx + dy * dy, 1.5);

		return (denom != 0.0) ? float(num / denom) : 0.f;

		// Evaluate r'(u) and r''(u)
		const std::vector<real_t> rp = d1.eval(static_cast<real_t>(u)).result();
		const std::vector<real_t> rpp = d2.eval(static_cast<real_t>(u)).result();

		const double eps = 1e-12;

		// 2D curvature
		if (dim == 2) {
			const double xp = static_cast<double>(rp[0]);
			const double yp = static_cast<double>(rp[1]);
			const double xpp = static_cast<double>(rpp[0]);
			const double ypp = static_cast<double>(rpp[1]);

			const double den2 = xp * xp + yp * yp;
			const double den = std::pow(den2, 1.5);
			if (den < eps) return 0.0;

			const double num = std::abs(xp * ypp - yp * xpp);
			std::cout << u << " -> " << num / den << std::endl;
			return num / den;
		}

		// 3D curvature
		// kappa = ||r' cross r''|| / ||r'||^3
		const double rp_norm = norm(rp);
		if (rp_norm < eps) return 0.0;

		auto cp = cross3(rp, rpp);
		const double cp_norm = std::sqrt(cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2]);
		const double den = rp_norm * rp_norm * rp_norm;
		if (den < eps) return 0.0;

		return (float)(cp_norm / den);
	}

	// Convenience: curvature for each control point (evaluated at its Greville abscissa)
	inline std::vector<float> curvature_per_control(const tinyspline::BSpline& s)
	{
		// First and second derivative splines
		tinyspline::BSpline d1 = s.derive();
		tinyspline::BSpline d2 = d1.derive();

		std::cout << __LINE__ << ", dim = " << s.dimension() << std::endl;
		const auto xi = greville_abscissae(s);
		std::cout << __LINE__ << " " << xi.size() << std::endl;
		std::vector<float> kappa;
		kappa.reserve(xi.size());
		for (double u : xi) {
			kappa.push_back(curvature_at(s, d1, d2, u));
		}
		std::cout << __LINE__ << std::endl;
		return kappa;
	}
	*/

using real_t = tinyspline::real;

inline double sqr(double x) { return x * x; }

// Convert TinySpline result vector of length 2 or 3 to 3D double
inline void to3(const std::vector<real_t>& v, double out[3]) {
	if (v.size() == 2) { out[0] = (double)v[0]; out[1] = (double)v[1]; out[2] = 0.0; }
	else if (v.size() == 3) { out[0] = (double)v[0]; out[1] = (double)v[1]; out[2] = (double)v[2]; }
	else throw std::runtime_error("Curvature only supported for dim 2 or 3.");
}

inline double norm3(const double a[3]) {
	return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline void cross3(const double a[3], const double b[3], double out[3]) {
	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}

// Compute Greville abscissae for a clamped or open B-spline.
// For control point i in [0..n-1], xi = (t_{i+1} + ... + t_{i+p}) / p where p = degree.
inline std::vector<double> greville_abscissae(const tinyspline::BSpline& s)
{
	const int p = (int)s.degree();
	const int nCtrl = (int)s.numControlPoints();
	if (p <= 0) throw std::runtime_error("Degree must be >= 1 to compute Greville points.");

	const std::vector<real_t> knots = s.knots();
	if ((int)knots.size() != nCtrl + p + 1)
		throw std::runtime_error("Unexpected knot vector size.");

	std::vector<double> xi;
	xi.reserve(nCtrl);
	for (int i = 0; i < nCtrl; ++i) {
		double sum = 0.0;
		for (int j = 1; j <= p; ++j) sum += (double)knots[i + j];
		xi.push_back(sum / p);
	}

	// Clamp to the curve domain
	tinyspline::Domain dom = s.domain();
	const double umin = (double)dom.min();
	const double umax = (double)dom.max();
	for (double& u : xi) {
		if (u < umin) u = umin;
		if (u > umax) u = umax;
	}
	return xi;
}

// Core curvature computation at parameter u. Returns NaN if evaluation failed.
inline double curvature_core(const tinyspline::BSpline& s, double u)
{
	// Build derivative splines once per call
	tinyspline::BSpline d1 = s.derive();
	tinyspline::BSpline d2 = d1.derive();

	const std::vector<real_t> rp = d1.eval((real_t)u).result();  // r'(u)
	const std::vector<real_t> rpp = d2.eval((real_t)u).result();  // r''(u)

	double r1[3], r2[3], c[3];
	to3(rp, r1);
	to3(rpp, r2);

	const double r1n = norm3(r1);
	// Strong epsilon relative to data scale
	const double eps = 1e-14;

	if (!(r1n > eps)) return std::numeric_limits<double>::quiet_NaN();

	cross3(r1, r2, c);
	const double num = norm3(c);
	const double den = r1n * r1n * r1n;

	if (!(den > eps)) return std::numeric_limits<double>::quiet_NaN();

	const double kappa = num / den;
	if (!std::isfinite(kappa)) return std::numeric_limits<double>::quiet_NaN();
	return abs(kappa);
}

// Safer curvature with parameter nudging to avoid endpoints and tiny tangents.
inline float curvature_at(const tinyspline::BSpline& s, double u)
{
	tinyspline::Domain dom = s.domain();
	const double umin = (double)dom.min();
	const double umax = (double)dom.max();
	const double span = std::max(umax - umin, 1.0); // avoid zero
	const double du = 1e-7 * span;                  // small nudge relative to domain

	auto clamp = [&](double x) {
		if (x < umin) return umin;
		if (x > umax) return umax;
		return x;
		};

	// Try at u
	double k = curvature_core(s, clamp(u));
	if (std::isfinite(k)) return (float)k;

	// Try nudged values
	double k1 = curvature_core(s, clamp(u + du));
	if (std::isfinite(k1)) return (float)k1;

	double k2 = curvature_core(s, clamp(u - du));
	if (std::isfinite(k2)) return (float)k2;

	// If still not finite, give up gracefully
	return 0.f;
}

// Curvature per control point evaluated at its Greville abscissa (with safety)
inline std::vector<float> curvature_per_control(const tinyspline::BSpline& s)
{
	const auto xi = greville_abscissae(s);
	std::vector<float> kappa;
	kappa.reserve(xi.size());
	for (double u : xi) {
		kappa.push_back(curvature_at(s, u));
	}
	return kappa;
}
} // namespace splineutils

namespace poca::geometry {
	ObjectListPolygon::ObjectListPolygon(const float* _xs, const float* _ys, const float* _zs, 
		const std::vector <std::vector<Polygon_2>>& _polygons, const std::vector <uint32_t>& _locsAllObjects, 
		const std::vector <uint32_t>& _firstsLocs, const std::vector <uint32_t>& _linkTriangulationFacesToObjects) : ObjectListInterface("ObjectListPolygon", _locsAllObjects, _firstsLocs), m_polygons(_polygons), m_xs(_xs), m_ys(_ys), m_zs(_zs), m_linkTriangulationFacesToObjects(_linkTriangulationFacesToObjects)
	{
		generateFromPolygons();
	}

	ObjectListPolygon::ObjectListPolygon(const std::vector <Polygon_2>& _polygons) : ObjectListInterface("ObjectListPolygon")
	{
		for (const auto& poly : _polygons)
			m_polygons.push_back({ poly });
		generateFromPolygons();
	}


	ObjectListPolygon::ObjectListPolygon(const std::vector <std::vector<Polygon_2>>& _polygons) : ObjectListInterface("ObjectListPolygon"), m_polygons(_polygons)
	{
		generateFromPolygons();
	}

	ObjectListPolygon::ObjectListPolygon(std::vector <std::vector<Polygon_2>>::const_iterator _begin, std::vector <std::vector<Polygon_2>>::const_iterator _end) : ObjectListInterface("ObjectListPolygon"), m_polygons(_begin, _end)
	{
		generateFromPolygons();
	}

	ObjectListPolygon::ObjectListPolygon(const std::vector <std::vector <std::vector <poca::core::Vec3mf>>>& _allSegments) : ObjectListInterface("ObjectListPolygon")
	{
		m_polygons.resize(_allSegments.size());

		uint32_t curObj = 0, curOutline = 0;
		for (const auto& segments : _allSegments) {
			for (const auto& segment : segments) {
				std::vector < Point_2 > ptsTmp;
				for (const auto& pt : segment)
					ptsTmp.push_back(Point_2(pt.x(), pt.y()));
				m_polygons[curObj].emplace_back(ptsTmp.begin(), ptsTmp.begin() + ptsTmp.size());
				/*for (const auto& point : m_polygons[curObj].back().container()) {
					std::cout << point << std::endl;
				}*/
				if (m_polygons[curObj].back().is_clockwise_oriented()) {
					std::cout << "Object " << curObj << ", cur outline " << m_polygons[curObj].size() << std::endl;
					m_polygons[curObj].back() = Polygon_2(ptsTmp.rbegin(), ptsTmp.rbegin() + ptsTmp.size());
				}
			}
			curObj++;
		}

		generateFromPolygons();
	}

	void ObjectListPolygon::generateFromPolygons()
	{
		std::vector <float> area;

		m_centroids.resize(m_polygons.size());
		m_bboxMeshes.resize(m_polygons.size());

		m_bbox.set(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0.f, std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), 0.f);
		uint32_t curObj = 0;
		for (const auto& polygons : m_polygons) {
			const auto& polygon = polygons.front();
			auto centroid = CGAL::centroid(polygon.vertices_begin(), polygon.vertices_end());
			auto bbox = CGAL::bounding_box(polygon.vertices_begin(), polygon.vertices_end());
			m_centroids[curObj].set(centroid.x(), centroid.y(), 0.f);
			m_bboxMeshes[curObj++].set(bbox.xmin(), bbox.ymin(), 0.f, bbox.xmax(), bbox.ymax(), 0.f);
			m_bbox.addPointBBox(bbox.xmin(), bbox.ymin(), 0.f);
			m_bbox.addPointBBox(bbox.xmax(), bbox.ymax(), 0.f);
		}

		if (m_locs.empty()) {
			std::vector <uint32_t> points, nbPts{ 0 }; //_mesh.number_of_vertices()
			uint32_t curObj = 0;
			for (const auto& polygons : m_polygons) {
				const auto& polygon = polygons.front();
				for (const auto& p : polygon.container()) {
					auto x = p.x(), y = p.y();
					m_xsDuplicate.push_back(x);
					m_ysDuplicate.push_back(y);
				}
				nbPts.push_back(m_xsDuplicate.size());
			}
			m_xs = m_xsDuplicate.data();
			m_ys = m_ysDuplicate.data();
			m_zs = NULL;

			points.resize(nbPts.back());
			std::iota(std::begin(points), std::end(points), 0);
			m_locs.initialize(points, nbPts);
			m_outlineLocs = m_locs;
		}

		//compute curvatures
		size_t cur = 0, nb = m_polygons.size();
		for (const auto& polygons : m_polygons) {
			try {
				m_curvatures.push_back(std::vector<std::vector <float>>());
				auto& allCurvatures = m_curvatures.back();
				size_t curS = 0, nbS = polygons.size();
				for (const auto& polygon : polygons) {
					std::cout << "polygon " << cur << " / " << nb << ", spline " << curS << " / " << nbS << ", # points" << polygon.size() << std::endl;
					/*std::vector<tinyspline::real> points;
					for (const auto& p : polygon.container()) {
						points.emplace_back(p.x());
						points.emplace_back(p.y());
					}
					tinyspline::BSpline spline = tinyspline::BSpline(points.size() / 2);
					auto curvs = splineutils::curvature_per_control(spline);
					curS++;
					auto [min_val, max_val] = percentile_bounds_2_98(curvs);
					float range = max_val - min_val;
					for (float& v : curvs) v = std::clamp(v, min_val, max_val);
					std::cout << "min curv " << min_val << ", max curv " << max_val << std::endl;
					std::transform(curvs.begin(), curvs.end(), curvs.begin(), [min_val, range](float val) { return (val - min_val) / range; });
					allCurvatures.push_back(curvs);*/
					auto ks = control_polygon_curvatures<K_inexact>(polygon);
					std::vector<float> abs_vals;
					abs_vals.reserve(ks.size());
					std::transform(ks.begin(), ks.end(), std::back_inserter(abs_vals), [](const auto& pr) { return pr.second; });
					/*auto [min_val, max_val] = percentile_bounds_2_98(abs_vals);
					float range = max_val - min_val;
					for (float& v : abs_vals) v = std::clamp(v, min_val, max_val);
					std::cout << "min curv " << min_val << ", max curv " << max_val << std::endl;
					std::transform(abs_vals.begin(), abs_vals.end(), abs_vals.begin(), [min_val, range](float val) { return (val - min_val) / range; });*/
					allCurvatures.push_back(abs_vals);
				}
			}
			catch (const std::runtime_error& e) {
				std::cerr << "Caught runtime_error: " << e.what() << std::endl;
			}
			cur++;
		}


		std::vector <poca::core::Vec3mf> outlines;
		std::vector <uint32_t> nbSegments{ 0 }; //_mesh.number_of_vertices()
		std::vector <float> curvOutlines;
		//for (const auto& polygons : m_polygons) {
		for (auto const& t1 : boost::combine(m_polygons, m_curvatures)) {
			auto const& polygons = t1.get<0>();
			auto& curvatures = t1.get<1>();
			std::cout << "****************************************" << polygons.size() << " vs " << curvatures.size() << std::endl;
			//for (const auto& polygon : polygons) {
			for (auto const& t2 : boost::combine(polygons, curvatures)) {
				auto const& polygon = t2.get<0>();
				auto& curvature = t2.get<1>();
				//std::cout << polygon.size() << " vs " << curvature.size() << std::endl;
				//std::cout << "Aera = " << fabs(polygon.area()) << ", # verts = " << polygon.size() << std::endl;
				const auto& points = polygon.container();
				std::size_t n = points.size();

				for (std::size_t i = 0; i < n; ++i) {
					const auto& curr = points[i];
					const auto& next = points[(i + 1) % n];  // wrap around

					outlines.emplace_back(curr.x(), curr.y(), 0.f);
					outlines.emplace_back(next.x(), next.y(), 0.f);

					curvOutlines.emplace_back(curvature[i]);
					curvOutlines.emplace_back(curvature[(i + 1) % n]);
				}
			}
			nbSegments.push_back(outlines.size());
		}
		m_outlines.initialize(outlines, nbSegments);
		//float min_val = *std::min_element(curvOutlines.begin(), curvOutlines.end());
		//float max_val = *std::max_element(curvOutlines.begin(), curvOutlines.end());
		auto [min_val, max_val] = poca::geometry::percentile_bounds_2_98(curvOutlines);
		float range = max_val - min_val;
		for (float& v : curvOutlines) v = std::clamp(v, min_val, max_val);
		std::cout << "min curv " << min_val << ", max curv " << max_val << std::endl;
		std::transform(curvOutlines.begin(), curvOutlines.end(), curvOutlines.begin(), [min_val, range](float val) { return (val - min_val) / range; });
		m_curvaturesArray.initialize(curvOutlines, nbSegments);
		m_minCurvature = 0.f;
		m_maxCurvature = 1.f;
		m_hasCurvature = true;
		//for (const auto f : curvOutlines)
		//	std::cout << f << ", ";
		//std::cout << std::endl;

		std::cout << "min curv " << m_minCurvature << ", max curv " << m_maxCurvature << std::endl;
		//std::cout << "****************************************" << std::endl;

		std::vector <poca::core::Vec3mf> triangles;
		std::vector <uint32_t> nbTriangles{ 0 }; //_mesh.number_of_vertices()
		if (false) {
			for (const auto& polygons : m_polygons) {
				const auto& polygon = polygons.size() >= 2 ? polygons[1] : polygons[0];
					Constrained_Delaunay_triangulation_2_tag cdt;
					auto insert_polygon = [&cdt](const Polygon_2& poly) {
						for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it) {
							auto next = std::next(it);
							if (next == poly.vertices_end()) next = poly.vertices_begin();
							cdt.insert_constraint(*it, *next);
						}
						};

					insert_polygon(polygon);

					size_t currentIndex = 0;
					for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
						Point_2 p0 = fit->vertex(0)->point();
						Point_2 p1 = fit->vertex(1)->point();
						Point_2 p2 = fit->vertex(2)->point();

						// Use centroid to check whether triangle is inside area of interest
						Point_2 centroid((p0.x() + p1.x() + p2.x()) / 3,
							(p0.y() + p1.y() + p2.y()) / 3);

						bool inside = polygon.bounded_side(centroid) == CGAL::ON_BOUNDED_SIDE;

						if (inside) {
							triangles.emplace_back(p0.x(), p0.y(), 0.f);
							triangles.emplace_back(p1.x(), p1.y(), 0.f);
							triangles.emplace_back(p2.x(), p2.y(), 0.f);
							fit->info().m_index = currentIndex++;
						}
						fit->info().m_tag = inside ? poca::geometry::INSIDE : poca::geometry::OUTSIDE;
					}
				nbTriangles.push_back(triangles.size());
				//std::cout << "Polygon, nb triangles " << triangles.size() << std::endl;
			}
		}
		else {
			for (const auto& polygons : m_polygons) {
				//std::cout << "****************************************\n# polygons " << polygons.size() << std::endl;

				for(const auto& pol : polygons)
					std::cout << "Polygon, # locs " << pol.size() << ", area " << pol.area() << std::endl;

				Polygon_with_holes_2_inexact pwh(polygons.front());
				for (auto n = 1; n < polygons.size(); n++)
					pwh.add_hole(polygons[n]);

				//std::cout << "-";
				m_cdts.push_back(Constrained_Delaunay_triangulation_2_tag());
				auto& cdt = m_cdts.back();
				//Constrained_Delaunay_triangulation_2 cdt;

				//std::cout << "-";
				auto insert_polygon = [&cdt](const Polygon_2& poly) {
					for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it) {
						auto next = std::next(it);
						if (next == poly.vertices_end()) next = poly.vertices_begin();
						cdt.insert_constraint(*it, *next);
					}
					};
				//std::cout << "-";

				insert_polygon(pwh.outer_boundary());
				for (auto hit = pwh.holes_begin(); hit != pwh.holes_end(); ++hit) {
					insert_polygon(*hit);
				}
				//std::cout << "-";

				// Optional: refine triangulation
				//CGAL::make_conforming_Delaunay_2(cdt);
				//CGAL::make_conforming_Gabriel_2(cdt);

				// 5. Collect valid triangles (inside outer, outside holes)
				//std::cout << "\nnumber of faces " << cdt.number_of_faces() << ", # vertices = " << cdt.number_of_vertices() << std::endl;
				size_t currentIndex = 0;
				for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
					Point_2 p0 = fit->vertex(0)->point();
					Point_2 p1 = fit->vertex(1)->point();
					Point_2 p2 = fit->vertex(2)->point();

					// Use centroid to check whether triangle is inside area of interest
					Point_2 centroid((p0.x() + p1.x() + p2.x()) / 3,
						(p0.y() + p1.y() + p2.y()) / 3);

					/*bool inside = pwh.outer_boundary().bounded_side(centroid) == CGAL::ON_BOUNDED_SIDE;

					if (inside) {
						triangles.emplace_back(p0.x(), p0.y(), 0.f);
						triangles.emplace_back(p1.x(), p1.y(), 0.f);
						triangles.emplace_back(p2.x(), p2.y(), 0.f);
						fit->info().m_index = currentIndex++;
					}
					fit->info().m_tag = inside ? poca::geometry::INSIDE : poca::geometry::OUTSIDE;*/



					// Check that centroid is INSIDE outer boundary and OUTSIDE all holes
					if (pwh.outer_boundary().bounded_side(centroid) == CGAL::ON_BOUNDED_SIDE) {
						bool inside_hole = false;
						for (auto hit = pwh.holes_begin(); hit != pwh.holes_end(); ++hit) {
							if (hit->bounded_side(centroid) == CGAL::ON_BOUNDED_SIDE) {
								inside_hole = true;
								break;
							}
						}
						if (!inside_hole) {
							triangles.emplace_back(p0.x(), p0.y(), 0.f);
							triangles.emplace_back(p1.x(), p1.y(), 0.f);
							triangles.emplace_back(p2.x(), p2.y(), 0.f);
							fit->info().m_index = currentIndex++;
						}
						fit->info().m_tag = inside_hole ? poca::geometry::OUTSIDE : poca::geometry::INSIDE;
					}
					else
						fit->info().m_tag = poca::geometry::OUTSIDE;
				}
				nbTriangles.push_back(triangles.size());
				//std::cout << "Polygon, nb triangles " << triangles.size() << std::endl;
				//for (const auto& pol : polygons) {
				//	std::cout << "\t area -> " << fabs(pol.area()) << std::endl;
				//}

			}
		}
		m_triangles.initialize(triangles, nbTriangles);

		//Create area feature
		std::vector <float> nbLocs(m_locs.nbElements(), 0.f);
		for (size_t i = 0; i < m_triangles.nbElements(); i++)
			nbLocs[i] = m_locs.nbElementsObject(i);
		const poca::core::MyArrayUInt32& localizations = m_locs;// m_outlineLocs;
		ObjectFeaturesFactoryInterface* factory = createObjectFeaturesFactory();
		std::vector <float> sizes(m_locs.nbElements()), resPCA(factory->nbFeaturesPCA(false));
		std::vector <float> circ(m_locs.nbElements()), areas(m_locs.nbElements());
		m_axis.resize(m_locs.nbElements());
		for (size_t n = 0; n < m_locs.nbElements(); n++) {
			float* ptr = &resPCA[0];
			factory->computePCA(m_locs, n, m_xs, m_ys, m_zs, ptr);
			sizes[n] = (resPCA[8] + resPCA[9]) / 2.f;
			circ[n] = resPCA[7];
			areas[n] = fabs(m_polygons[n].front().area());
			for(auto i = 1; i < m_polygons[n].size(); i++)
				areas[n] -= fabs(m_polygons[n][i].area());;
		}
		delete factory;

		std::vector <float> ids(m_locs.nbElements());
		std::iota(std::begin(ids), std::end(ids), 1);

		m_data["area"] = poca::core::generateDataWithLog(areas);
		m_data["nbLocs"] = poca::core::generateDataWithLog(nbLocs);
		m_data["size"] = poca::core::generateDataWithLog(sizes);
		m_data["circ"] = poca::core::generateDataWithLog(circ);
		m_data["id"] = poca::core::generateDataWithLog(ids);

		m_selection.resize(areas.size());
		setCurrentHistogramType("area");
		forceRegenerateSelection();
	}

	ObjectListPolygon::~ObjectListPolygon()
	{
	}

	poca::core::BasicComponentInterface* ObjectListPolygon::copy()
	{
		return new ObjectListPolygon(*this);
	}

	poca::core::BasicComponentInterface* ObjectListPolygon::copy(const std::vector <poca::core::ROIInterface*>& _ROIs)
	{
		if (_ROIs.empty())
			return copy();

		std::vector <std::vector <std::vector <poca::core::Vec3mf>>> allVertices;

		for (const auto& polygon : m_polygons) {
			bool inside = false;
			for (const auto& outline : polygon) {
				for (const auto& p : outline.container()) {
					for (auto curROI = 0; curROI < _ROIs.size() && !inside; curROI++) {
						inside = _ROIs[curROI]->inside(p.x(), p.y(), 0.f);
						if (inside) break;
					}
					if (inside) break;
				}
			}
			if (inside) {
				allVertices.push_back(std::vector < std::vector <poca::core::Vec3mf>>());
				auto& vertices = allVertices.back();
				for (const auto& outline : polygon) {
					vertices.push_back(std::vector<poca::core::Vec3mf>());
					for (const auto& p : outline.container()) {
						vertices.back().emplace_back(p.x(), p.y(), 0.f);
					}
				}
			}
		}
		return new ObjectListPolygon(allVertices);
	}

	poca::geometry::ObjectListPolygon* ObjectListPolygon::exportFilteredObjects()
	{
		std::vector <std::vector <std::vector <poca::core::Vec3mf>>> allVertices;

		for (auto n = 0; n < m_polygons.size(); n++) {
			if (m_selection[n]) {
				const auto& polygon = m_polygons[n];
				allVertices.push_back(std::vector < std::vector <poca::core::Vec3mf>>());
				auto& vertices = allVertices.back();
				for (const auto& outline : polygon) {
					vertices.push_back(std::vector<poca::core::Vec3mf>());
					for (const auto& p : outline.container()) {
						vertices.back().emplace_back(p.x(), p.y(), 0.f);
					}
				}
			}
		}
		return allVertices.empty() ? NULL: new ObjectListPolygon(allVertices);
	}
	
	void ObjectListPolygon::generateLocs(std::vector <poca::core::Vec3mf>& _locs)
	{
		/*_locs.clear();
		for (const auto& mesh : m_meshes)
			for (const auto& point : mesh.points())
				_locs.push_back(poca::core::Vec3mf(point.x(), point.y(), point.z()));*/
		_locs.resize(m_locs.nbData());
		const std::vector <uint32_t>& indices = m_locs.getData();
		for (size_t n = 0; n < indices.size(); n++) {
			size_t index = indices.at(n);
			_locs[n].set(m_xs[index], m_ys[index], 0.f);
		}
	}

	void ObjectListPolygon::generateNormalLocs(std::vector <poca::core::Vec3mf>& _norms)
	{
		_norms.resize(m_locs.nbData());
		const std::vector <uint32_t>& indices = m_locs.getData(), & objects = m_locs.getFirstElements();
		for (uint32_t n = 0; n < m_locs.nbElements(); n++) {
			uint32_t nbLocs = objects[n + 1] - objects[n];
			float nbD = nbLocs;
			poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
			for (uint32_t idx = objects[n]; idx < objects[n + 1]; idx++) {
				uint32_t index = indices[idx];
				centroid += poca::core::Vec3mf(m_xs[index], m_ys[index], 0.f) / nbD;
			}
			for (uint32_t idx = objects[n]; idx < objects[n + 1]; idx++) {
				uint32_t index = indices[idx];
				poca::core::Vec3mf normal = poca::core::Vec3mf(m_xs[index], m_ys[index], 0.f) - centroid;
				normal.normalize();
				_norms[idx] = normal;
			}
		}
	}

	void ObjectListPolygon::getLocsFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::getLocsFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		_features.resize(m_locs.nbData());// *2);

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::getOutlinesFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		/*_features.resize(m_outlines.nbData());// *2);

		size_t cpt = 0;
		for (size_t i = 0; i < m_outlines.nbElements(); i++) {
			for (size_t j = 0; j < m_outlines.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}
		}*/

		std::cout << "getOutlinesFeatureInSelection " << m_curvaturesArray.nbData() << std::endl;
		_features.resize(m_curvaturesArray.nbData());// *2);

		size_t cpt = 0;
		for (size_t i = 0; i < m_curvaturesArray.nbElements(); i++) {
			for (size_t j = 0; j < m_curvaturesArray.nbElementsObject(i); j++) {
				_features[cpt++] = m_curvaturesArray.elementIObject(i, j);// _selection[i] ? m_curvaturesArray.elementIObject(i, j) : _notSelectedValue;
			}
		}
	}

	void ObjectListPolygon::getOutlinesFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		_features.resize(m_outlines.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_outlines.nbElements(); i++) {
			for (size_t j = 0; j < m_outlines.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::generateLocsPickingIndices(std::vector <float>& _ids) const
	{
		/*_ids.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_ids.push_back(i + 1);
			i++;
		}*/
		_ids.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_ids[cpt++] = i + 1;
			}

		}
	}

	void ObjectListPolygon::generateTriangles(std::vector <poca::core::Vec3mf>& _triangles)
	{
		std::copy(m_triangles.getData().begin(), m_triangles.getData().end(), std::back_inserter(_triangles));
	}

	void ObjectListPolygon::generateOutlines(std::vector <poca::core::Vec3mf>& _outlines)
	{
		std::copy(m_outlines.getData().begin(), m_outlines.getData().end(), std::back_inserter(_outlines));
	}

	void ObjectListPolygon::generateNormals(std::vector <poca::core::Vec3mf>& _normals)
	{
		_normals.resize(m_triangles.getData().size());
		std::fill(_normals.begin(), _normals.end(), poca::core::Vec3mf(0.f, 0.f, 1.f));
	}

	void ObjectListPolygon::generatePickingIndices(std::vector <float>& _ids) const
	{
		const std::vector <poca::core::Vec3mf> triangles = m_triangles.getData();
		_ids.resize(triangles.size());

		size_t cpt = 0;
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j++) {
				_ids[cpt++] = i + 1;
			}

		}
	}

	void ObjectListPolygon::getFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		const std::vector <poca::core::Vec3mf> triangles = m_triangles.getData();
		_features.resize(triangles.size());

		size_t cpt = 0;
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::getFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		const std::vector <poca::core::Vec3mf> triangles = m_triangles.getData();
		_features.resize(triangles.size());

		size_t cpt = 0;
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	poca::core::BoundingBox ObjectListPolygon::computeBoundingBoxElement(const int _idx) const
	{
		return m_bboxMeshes[_idx];
	}

	poca::core::Vec3mf ObjectListPolygon::computeBarycenterElement(const int _idx) const
	{
		return m_centroids[_idx];
	}

	void ObjectListPolygon::generateOutlineLocs(std::vector <poca::core::Vec3mf>& _locs)
	{
		/*_locs.clear();
		for (const auto& mesh : m_meshes)
			for (const auto& point : mesh.points())
				_locs.push_back(poca::core::Vec3mf(point.x(), point.y(), point.z()));*/
		_locs.resize(m_locs.nbData());
		const std::vector <uint32_t>& indices = m_locs.getData();
		for (size_t n = 0; n < indices.size(); n++) {
			size_t index = indices.at(n);
			_locs[n].set(m_xs[index], m_ys[index], 0.f);
		}
	}

	void ObjectListPolygon::getOutlineLocsFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		/*_features.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_features.push_back(_selection[i] ? _values[i] : _notSelectedValue);
			i++;
		}*/
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::getOutlineLocsFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		/*_features.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_features.push_back(_selection[i] ? _selectedValue : _notSelectedValue);
			i++;
		}*/
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectListPolygon::saveAsPol(const std::string& _filename) const
	{
		std::ofstream fs(_filename, std::ifstream::binary);
		size_t nb = m_polygons.size();
		fs.write(reinterpret_cast<char*>(&nb), sizeof(size_t));
		for (const auto& polygons : m_polygons) {
			nb = polygons.size();
			fs.write(reinterpret_cast<char*>(&nb), sizeof(size_t));

			for (const auto& poly : polygons) {
				std::vector <poca::core::Vec3mf> vertices;
				for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it)
					vertices.emplace_back(it->x(), it->y(), 0.f);
				nb = vertices.size();
				fs.write(reinterpret_cast<char*>(&nb), sizeof(size_t));
				fs.write(reinterpret_cast<char*>(vertices.data()), nb * sizeof(poca::core::Vec3mf));
			}
		}
		fs.close();
	}

	poca::geometry::ObjectListInterface* ObjectListPolygon::exportFilteredObjects() const
	{
		std::vector <std::vector<Polygon_2>> polygonsSelected;

		for (auto n = 0; n < m_polygons.size(); n++) {
			if (m_selection[n])
				polygonsSelected.push_back(m_polygons[n]);
		}
		return polygonsSelected.empty() ? NULL : static_cast <poca::geometry::ObjectListInterface*> (new ObjectListPolygon(polygonsSelected));
	}
}
