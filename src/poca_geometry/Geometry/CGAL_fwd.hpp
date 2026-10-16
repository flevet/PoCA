/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#pragma once

// Exact aliases from CGAL_includes.hpp, for pointer/reference declarations only.
// Include CGAL_includes.hpp where complete point/mesh definitions are needed.
namespace CGAL {
	template<typename FT_> struct Simple_cartesian;
	template<typename R_> class Point_3;
	template<typename P> class Surface_mesh;
}
using Kernel = CGAL::Simple_cartesian<double>;
using Point_3_double = CGAL::Point_3<Kernel>;
using Surface_mesh_3_double = CGAL::Surface_mesh<Point_3_double>;
