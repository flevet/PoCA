/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ArrayStatistics.cpp
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

#include <vector>
#include <algorithm>
#include <execution>

#include "ArrayStatistics.hpp"

namespace poca::core {

	ArrayStatistics::ArrayStatistics()
	{ 
		m_data.resize(STATS_NB_PARAMS, 0.f);
	}

	ArrayStatistics::ArrayStatistics(const float _mean, const float _median, const float _stdDev, const float _min, const float _max) 
	{
		m_data[Mean] = _mean;
		m_data[Median] = _median;
		m_data[StdDev] = _stdDev;
		m_data[Min] = _min; 
		m_data[Max] = _max;
	}

	ArrayStatistics::ArrayStatistics(const std::vector<float>& _vals) 
	{ 
		std::copy(_vals.begin(), _vals.end(), std::back_inserter(m_data));
	}

	/*ArrayStatistics ArrayStatistics::generateArrayStatistics(const float* _data, const size_t _nb)
	{
		float nb = (float)_nb;
		if (_nb == 0) return ArrayStatistics();
		std::vector <float> vals(STATS_NB_PARAMS);
		vals[ArrayStatistics::Mean] = vals[ArrayStatistics::Median] = vals[ArrayStatistics::StdDev] = 0.;
		vals[ArrayStatistics::Min] = FLT_MAX;
		vals[ArrayStatistics::Max] = -FLT_MAX;


		std::vector < float > vectMedian(_nb);
		for (size_t n = 0; n < _nb; n++) {
			vals[ArrayStatistics::Mean] += (_data[n] / nb);
			vectMedian[n] = _data[n];
			if (_data[n] > vals[ArrayStatistics::Max])
				vals[ArrayStatistics::Max] = _data[n];
			if (_data[n] < vals[ArrayStatistics::Min])
				vals[ArrayStatistics::Min] = _data[n];
		}
		float nbForDev = nb - 1.f;
		for (size_t n = 0; n < _nb; n++) {
			float deviation = vectMedian[n] - vals[ArrayStatistics::Mean];
			vals[ArrayStatistics::StdDev] += ((deviation * deviation) / nbForDev);
		}
		vals[ArrayStatistics::StdDev] = sqrt(vals[ArrayStatistics::StdDev]);
		std::sort(vectMedian.begin(), vectMedian.end());
		vals[ArrayStatistics::Median] = vectMedian[vectMedian.size() / 2];

		ArrayStatistics stats(vals);
		return stats;
	}*/

	/*ArrayStatistics ArrayStatistics::generateInverseArrayStatistics(const std::vector <float>& _data, const size_t _nb)
	{
		return ArrayStatistics::generateInverseArrayStatistics(_data.data(), _nb);
	}

	ArrayStatistics ArrayStatistics::generateInverseArrayStatistics(const float* _invData, const size_t _nb)
	{
		float nbData = 0.;
		if (_nb == 0) return ArrayStatistics();
		float vals[STATS_NB_PARAMS];
		vals[ArrayStatistics::Mean] = vals[ArrayStatistics::Median] = vals[ArrayStatistics::StdDev] = 0.;
		vals[ArrayStatistics::Min] = FLT_MAX;
		vals[ArrayStatistics::Max] = -FLT_MAX;
		for (int n = 0; n < _nb; n++)
			nbData += _invData[n];
		std::vector < float > vectMedian((int)nbData);
		std::vector < float >::iterator it = vectMedian.begin();
		for (size_t n = 0; n < _nb; n++) {
			vals[ArrayStatistics::Mean] += (((float)n * _invData[n]) / nbData);
			std::fill_n(it, (int)_invData[n], n);
			it += _invData[n];
			if (_invData[n] > 0 && _invData[n] > vals[ArrayStatistics::Max])
				vals[ArrayStatistics::Max] = _invData[n];
			if (_invData[n] > 0 && _invData[n] < vals[ArrayStatistics::Min])
				vals[ArrayStatistics::Min] = _invData[n];
		}

		float nbForDev = nbData - 1.f;
		for (size_t n = 0; n < nbData; n++) {
			float deviation = vectMedian[n] - vals[ArrayStatistics::Mean];
			vals[ArrayStatistics::StdDev] += ((deviation * deviation) / nbForDev);
		}
		vals[ArrayStatistics::StdDev] = sqrt(vals[ArrayStatistics::StdDev]);
		std::sort(vectMedian.begin(), vectMedian.end());
		vals[ArrayStatistics::Median] = vectMedian[vectMedian.size() / 2];

		ArrayStatistics stats(vals);
		return stats;
	}*/
	std::ostream& operator<<(std::ostream& _os, const ArrayStatistics& _stats)
	{
		return _os << "[Mean : " << _stats.getData(ArrayStatistics::Mean) << ", median : " << _stats.getData(ArrayStatistics::Median) << ", std dev : " << _stats.getData(ArrayStatistics::StdDev) << ", min : " << _stats.getData(ArrayStatistics::Min) << ", max = " << _stats.getData(ArrayStatistics::Max) << "]";
	}
}

