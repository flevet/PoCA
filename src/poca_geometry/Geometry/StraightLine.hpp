/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      StraightLine.hpp
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

#ifndef StraightLine_h__
#define StraightLine_h__

#include <General/Vec2.hpp>

namespace poca::geometry {
	class StraightLine {
	public:
		enum LineType { PARALLELE_LINE = 0, PERPENDICULAR_LINE = 1 };
		StraightLine(const double, const double, const double, const double, const int);

		const poca::core::Vec2md orthoProjection(const double, const double) const;
		const poca::core::Vec2md intersectionSegments(const StraightLine&, bool&) const;
		const poca::core::Vec2md intersectionOneSegment(const StraightLine&, bool&) const;
		const poca::core::Vec2md intersectionLine(const StraightLine&, bool&) const;
		const double eval(const double, const double) const;
		const double evalSign(const double, const double) const;
		const poca::core::Vec2md findPoint(const double, const double, const double) const;
		const poca::core::Vec2md findOrthoPoint(const double, const double, const double) const;
		const bool isOrthProjInRange(const double, const double) const;

		inline const poca::core::Vec2md& getP1() const { return m_p1; }
		inline const poca::core::Vec2md& getP2() const { return m_p2; }
		inline const double getA() const { return m_a; }
		inline const double getB() const { return m_b; }
		inline const double getC() const { return m_c; }

	protected:
		double m_a, m_b, m_c;
		int m_type;
		poca::core::Vec2md m_p1, m_p2;
	};
}
#endif

