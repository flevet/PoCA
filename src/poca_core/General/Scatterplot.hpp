/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Scatterplot.hpp
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

#ifndef Scatterplot_h__
#define Scatterplot_h__

#include "../Interfaces/ScatterplotInterface.hpp"

namespace poca::core {
	class Scatterplot : public ScatterplotInterface {
	public:
		Scatterplot(){}
		Scatterplot(const std::vector <poca::core::Vec2mf>& _pts) : m_points(_pts) {}
		~Scatterplot() {}

		const std::vector <poca::core::Vec2mf>& getPoints() const { return m_points; }
		std::vector <poca::core::Vec2mf>& getPoints() { return m_points; }

		const float getThresholdX() const { return m_thresholds[0]; }
		void setThresholdX(const float _val) { m_thresholds[0] = _val; }
		const float getThresholdY() const { return m_thresholds[1]; }
		void setThresholdY(const float _val) { m_thresholds[1] = _val; }

		poca::core::Vec2mf getThreshold() const { return m_thresholds; }
		void setThreshold(const poca::core::Vec2mf& _val) { m_thresholds = _val; }

		const std::size_t getNbValues() const { return m_points.size(); }
		const size_t size() const { return m_points.size(); }

		void resize(size_t _size) { m_points.resize(_size); }
		void set(size_t _index, const float _x, const float _y) { m_points[_index].set(_x, _y); }
		poca::core::Vec2mf& operator[](int _idx) { return m_points[_idx]; }
		const poca::core::Vec2mf& operator[](int _idx) const { return m_points[_idx]; }

	protected:
		std::vector <poca::core::Vec2mf> m_points;
		poca::core::Vec2mf m_thresholds;
	};
}
#endif

