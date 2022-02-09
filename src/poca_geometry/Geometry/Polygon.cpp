#include "Polygon.hpp"

namespace poca::geometry {
	Polygon::Polygon(const std::vector <poca::core::Vec2mf>& _pts)
	{
		std::vector < Point_2 > ptsTmp;
		for (const poca::core::Vec2mf& pt : _pts)
			ptsTmp.push_back(Point_2(pt.x(), pt.y()));
		m_poly = Polygon_2(ptsTmp.begin(), ptsTmp.begin() + ptsTmp.size());
		if (!m_poly.is_simple()) {
			std::cout << "Polygon not simple" << std::endl;
			m_poly.clear();
		}
		else
			if (m_poly.is_clockwise_oriented())
				m_poly = Polygon_2(ptsTmp.rbegin(), ptsTmp.rbegin() + ptsTmp.size());
	}

	Polygon::Polygon(const Polygon_2& _poly):m_poly(_poly)
	{
	}
}