/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      StraightLine.cpp
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

#include "StraightLine.hpp"

namespace poca::geometry {

	StraightLine::StraightLine(const double _x1, const double _y1, const double _x2, const double _y2, const int _type) :m_p1(_x1, _y1), m_p2(_x2, _y2), m_type(_type)
	{
		double vx = _x2 - _x1;
		double vy = _y2 - _y1;

		/*Nomalization of the vector director*/
		double length = sqrt(vx * vx + vy * vy);
		vx /= length;
		vy /= length;

		if (m_type == PERPENDICULAR_LINE) {
			m_a = -vy;
			m_b = vx;
		}
		else if (m_type == PARALLELE_LINE) {
			m_a = vx;
			m_b = vy;
		}
		m_c = -(m_a * _x2 + m_b * _y2);
	}

	const poca::core::Vec2md StraightLine::orthoProjection(const double _x, const double _y) const
	{
		double a = m_p2.y() - m_p1.y();
		double b = m_p1.x() - m_p2.x();
		double c = (m_p1.y() - m_p2.y()) * m_p1.x() + (m_p2.x() - m_p1.x()) * m_p1.y();
		double AA = (m_p2.x() - m_p1.x());
		double BB = (m_p2.y() - m_p1.y());
		double y = (-(a * _x) - (a * BB * _y / AA) - c) / (b - (a * BB / AA));
		double x = (-c - b * y) / a;
		return poca::core::Vec2md(x, y);
	}

	//Intersection point has to be inside the two segments
	const poca::core::Vec2md StraightLine::intersectionSegments(const StraightLine& _line, bool& _ok) const
	{
		poca::core::Vec2md intersection;
		_ok = false;
		if (m_a * _line.m_b - m_b * _line.m_a == 0)
			return intersection;
		double a = m_p2.x() - m_p1.x(), b = m_p2.y() - m_p1.y();
		double r = ((m_p1.y() - _line.m_p1.y()) * (_line.m_p2.x() - _line.m_p1.x()) - (m_p1.x() - _line.m_p1.x()) * (_line.m_p2.y() - _line.m_p1.y()));
		r /= ((m_p2.x() - m_p1.x()) * (_line.m_p2.y() - _line.m_p1.y()) - (m_p2.y() - m_p1.y()) * (_line.m_p2.x() - _line.m_p1.x()));
		double s = ((m_p1.y() - _line.m_p1.y()) * (m_p2.x() - m_p1.x()) - (m_p1.x() - _line.m_p1.x()) * (m_p2.y() - m_p1.y()));
		s /= ((m_p2.x() - m_p1.x()) * (_line.m_p2.y() - _line.m_p1.y()) - (m_p2.y() - m_p1.y()) * (_line.m_p2.x() - _line.m_p1.x()));
		if (0. <= r && r <= 1. && 0. <= s && s <= 1.) {
			_ok = true;
			intersection.set(m_p1.x() + r * a, m_p1.y() + r * b);
		}
		return intersection;
	}

	//intersection point has to be only inside the _line segment, the this segment is treated as a straightline
	const poca::core::Vec2md StraightLine::intersectionOneSegment(const StraightLine& _line, bool& _ok) const
	{
		poca::core::Vec2md intersection;
		_ok = false;
		if (m_a * _line.m_b - m_b * _line.m_a == 0)
			return intersection;
		double a = m_p2.x() - m_p1.x(), b = m_p2.y() - m_p1.y();
		double r = ((m_p1.y() - _line.m_p1.y()) * (_line.m_p2.x() - _line.m_p1.x()) - (m_p1.x() - _line.m_p1.x()) * (_line.m_p2.y() - _line.m_p1.y()));
		r /= ((m_p2.x() - m_p1.x()) * (_line.m_p2.y() - _line.m_p1.y()) - (m_p2.y() - m_p1.y()) * (_line.m_p2.x() - _line.m_p1.x()));
		double s = ((m_p1.y() - _line.m_p1.y()) * (m_p2.x() - m_p1.x()) - (m_p1.x() - _line.m_p1.x()) * (m_p2.y() - m_p1.y()));
		s /= ((m_p2.x() - m_p1.x()) * (_line.m_p2.y() - _line.m_p1.y()) - (m_p2.y() - m_p1.y()) * (_line.m_p2.x() - _line.m_p1.x()));
		if (0. <= s && s <= 1.) {
			_ok = true;
			intersection.set(m_p1.x() + r * a, m_p1.y() + r * b);
		}
		return intersection;
	}

	//intersection point does not need to be inside the two segments, they are treated as straightlines 
	const poca::core::Vec2md StraightLine::intersectionLine(const StraightLine& _line, bool& _ok) const
	{
		poca::core::Vec2md intersection;
		_ok = false;
		if (m_a * _line.m_b - m_b * _line.m_a == 0)
			return intersection;
		double a = m_p2.x() - m_p1.x(), b = m_p2.y() - m_p1.y();
		double r = ((m_p1.y() - _line.m_p1.y()) * (_line.m_p2.x() - _line.m_p1.x()) - (m_p1.x() - _line.m_p1.x()) * (_line.m_p2.y() - _line.m_p1.y()));
		r /= ((m_p2.x() - m_p1.x()) * (_line.m_p2.y() - _line.m_p1.y()) - (m_p2.y() - m_p1.y()) * (_line.m_p2.x() - _line.m_p1.x()));
		_ok = true;
		intersection.set(m_p1.x() + r * a, m_p1.y() + r * b);
		return intersection;
	}

	const double StraightLine::eval(const double _x, const double _y) const
	{
		return m_a * _x + m_b * _y + m_c;
	}

	const double StraightLine::evalSign(const double _x, const double _y) const
	{
		double val = eval(_x, _y);
		return val / abs(val);
	}

	const poca::core::Vec2md StraightLine::findPoint(const double _x, const double _y, const double _k) const
	{
		double x = _x + _k * m_a;
		double y = _y + _k * m_b;
		return poca::core::Vec2md(x, y);
	}

	const poca::core::Vec2md StraightLine::findOrthoPoint(const double _x, const double _y, const double _k) const
	{
		double x = _x + _k * -m_b;
		double y = _y + _k * m_a;
		return poca::core::Vec2md(x, y);
	}

	const bool StraightLine::isOrthProjInRange(const double _x, const double _y) const
	{
		double dx = m_p2.x() - m_p1.x();
		double dy = m_p2.y() - m_p1.y();
		double innerProduct = (_x - m_p1.x()) * dx + (_y - m_p1.y()) * dy;
		return 0 <= innerProduct && innerProduct <= dx * dx + dy * dy;
	}

}

