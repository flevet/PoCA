/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ArrayStatistics.hpp
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

#ifndef ArrayStatistics_h__
#define ArrayStatistics_h__

#include <iostream>
#include <vector>

namespace poca::core {

#define STATS_NB_PARAMS 5

	class ArrayStatistics {
	public:
		enum StatsParam { Mean = 0, Median = 1, StdDev = 2, Min = 3, Max = 4 };
		ArrayStatistics();
		ArrayStatistics(const float, const float, const float, const float, const float);
		ArrayStatistics(float[STATS_NB_PARAMS]);

		inline void setData(const int _type, const float _val) { m_data[_type] = _val; }
		inline const float getData(const int _type) const { return m_data[_type]; }

		static ArrayStatistics generateArrayStatistics(const std::vector<float>&, const size_t);
		static ArrayStatistics generateArrayStatistics(const float*, const size_t);
		static ArrayStatistics generateInverseArrayStatistics(const std::vector<float>&, const size_t);
		static ArrayStatistics generateInverseArrayStatistics(const float*, const size_t);

		friend std::ostream& operator<<(std::ostream&, const ArrayStatistics&);

	private:
		float m_data[5];
	};

}
#endif // GeneralTools_h__

