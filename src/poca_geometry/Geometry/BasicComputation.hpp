/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicComputation.hpp
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

#ifndef BasicComputation_h__
#define BasicComputation_h__

#include <General/Vec2.hpp>
#include <General/Vec3.hpp>

namespace poca::geometry {
	float computeTriangleArea(const double, const double, const double, const double, const double, const double);
	float computePolygonArea(poca::core::Vec3md*, const unsigned int);
	float computePolygonArea2D(poca::core::Vec3md*, const unsigned int);
	float computePolygonArea(poca::core::Vec2md*, const unsigned int);
	float computePolygonArea(double*, size_t);
	float distance(const float, const float, const float, const float, const float, const float);
	float distanceSqr(const float, const float, const float, const float, const float, const float);

	//Computed using the Heron's formula
	//Variables are the three side lengthes of the triangle
	template <typename T>
	static T computeAreaTriangle(const T _a, const T _b, const T _c) {
		T semiPerimeter = (_a + _b + _c) / 2.;
		return (T)sqrt(fabs(semiPerimeter * (semiPerimeter - _a) * (semiPerimeter - _b) * (semiPerimeter - _c)));
	}

	class BasicComputation {
	public:
		static double distance(const double, const double, const double, const double);
		static double distanceSqr(const double, const double, const double, const double);

		static double getTriangleArea(const poca::core::Vec2md&, const poca::core::Vec2md&, const poca::core::Vec2md&);
		static double getTriangleArea(const double, const double, const double, const double, const double, const double);

		static bool isRecInsideCircle(const double, const double, const double, const double, const double, const double, const double);
		static bool isInsideRec(const double, const double, const double, const double, const double, const double);
		static bool isLineIntersectCircle(const double, const double, const double, const double, const double, const double, const double);
		static void closestPointOnLine(const double, const double, const double, const double, const double, const double, double&, double&);
		static bool circleLineIntersect(const double, const double, const double, const double, const double, const double, const double);

		static void circleLineIntersect(const double, const double, const double, const double, const double, const double, const double, std::vector < poca::core::Vec2md >&);
		static double computeAreaCircularSegment(const double, const double, const double, const poca::core::Vec2md&, const poca::core::Vec2md&);

	};
}
#endif

