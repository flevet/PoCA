/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Histogram.cpp
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

#include <float.h>
#include <iostream>

#include "Histogram.hpp"

namespace poca::core {
	Histogram::Histogram():m_nbValues(0), m_nbBins(0), m_stepX(0.f), m_maxY(FLT_MAX), m_currentMin(-FLT_MAX), m_currentMax(FLT_MAX), m_isMinDefined(false), m_isMaxDefined(false), m_isLog(false), m_minDefined(0.f), m_maxDefined(0.f)
	{
	}

	Histogram::Histogram(const std::vector <float>& _vals, const std::size_t _nbVals, const bool _isLog, const int _nbBins, const bool _isMinDefined, const float _minDefined, const bool _isMaxDefined, const float _maxDefined)
	{
		setHistogram(_vals.data(), _nbVals, _isLog, _nbBins, _isMinDefined, _minDefined, _isMaxDefined, _maxDefined);
	}

	Histogram::Histogram(const float* _vals, const std::size_t _nbVals, const bool _isLog, const int _nbBins, const bool _isMinDefined, const float _minDefined, const bool _isMaxDefined, const float _maxDefined)
	{
		setHistogram(_vals, _nbVals, _isLog, _nbBins, _isMinDefined, _minDefined, _isMaxDefined, _maxDefined);
	}

	Histogram::Histogram(const Histogram& _o) :m_values(_o.m_values), m_bins(_o.m_bins), m_ts(_o.m_ts), m_nbValues(_o.m_nbValues), m_nbBins(_o.m_nbBins),
		m_stats(_o.m_stats), m_stepX(_o.m_stepX), m_maxY(_o.m_maxY), m_currentMin(_o.m_currentMin),
		m_currentMax(_o.m_currentMax), m_isMinDefined(_o.m_isMinDefined), m_isMaxDefined(_o.m_isMaxDefined), m_isLog(_o.m_isLog),
		m_minDefined(_o.m_minDefined), m_maxDefined(_o.m_maxDefined)
	{
	}

	Histogram& Histogram::operator=(const Histogram& _o)
	{
		m_values = _o.m_values;
		m_bins = _o.m_bins;
		m_ts = _o.m_ts;
		m_nbValues = _o.m_nbValues;
		m_nbBins = _o.m_nbBins;
		m_stats = _o.m_stats;
		m_stepX = _o.m_stepX;
		m_maxY = _o.m_maxY;
		m_currentMin = _o.m_currentMin;
		m_currentMax = _o.m_currentMax;
		m_isMinDefined = _o.m_isMinDefined;
		m_isMaxDefined = _o.m_isMaxDefined;
		m_isLog = _o.m_isLog;
		m_minDefined = _o.m_minDefined;
		m_maxDefined = _o.m_maxDefined;

		return *this;
	}

	Histogram::~Histogram()
	{
	}

	void Histogram::setHistogram(const float* _vals, const std::size_t _nbVals, const bool _isLog, const int _nbBins, const bool _isMinDefined, const float _minDefined, const bool _isMaxDefined, const float _maxDefined)
	{
		m_isMinDefined = _isMinDefined; m_isMaxDefined = _isMaxDefined; m_minDefined = _minDefined; m_maxDefined = _maxDefined; m_isLog = _isLog;

		m_nbValues = _nbVals;
		m_values.resize(m_nbValues);
		std::copy(_vals, _vals + _nbVals, m_values.begin());

		m_stats = ArrayStatistics::generateArrayStatistics(m_values, m_nbValues);
		m_currentMin = m_stats.getData(ArrayStatistics::Min);
		m_currentMax = m_stats.getData(ArrayStatistics::Max);

		setNbBins(_nbBins);
	}

	void Histogram::changeHistogramBounds(const float _min, const float _max)
	{
		m_isMinDefined = _min != FLT_MAX;
		m_isMaxDefined = _min != FLT_MAX;
		if (m_minDefined || m_isMaxDefined) {
			m_minDefined = m_isMinDefined ? _min : m_minDefined;
			m_maxDefined = m_isMaxDefined ? _max : m_maxDefined;
			setNbBins(m_nbBins);
		}
	}

	void Histogram::eraseBounds()
	{
		m_currentMin = FLT_MIN;
		m_currentMax = FLT_MAX;
	}


	void Histogram::resetBounds()
	{
		m_currentMin = m_stats.getData(ArrayStatistics::Min);
		m_currentMax = m_stats.getData(ArrayStatistics::Max);
	}

	void Histogram::setNbBins(const std::size_t _nbBins)
	{
		setNbBins(_nbBins, m_values);
	}

	void Histogram::setNbBins(const std::size_t _nbBins, const std::vector <float>& _values)
	{
		setNbBins(_nbBins, m_values.data(), m_values.size());
	}

	void Histogram::setNbBins(const std::size_t _nbBins, const float* _values, const std::size_t _nbValues)
	{
		m_nbBins = _nbBins;

		m_bins.resize(m_nbBins, 0.);
		m_ts.resize(m_nbBins);

		float minTemp = (m_isMinDefined) ? m_minDefined : m_stats.getData(ArrayStatistics::Min);
		float maxTemp = (m_isMaxDefined) ? m_maxDefined : m_stats.getData(ArrayStatistics::Max);
		m_stepX = (maxTemp - minTemp) / (float)(m_nbBins - 1);
		for (unsigned int i = 0; i < _nbValues; i++) {
			if (_values[i] == -1) continue;
			unsigned short index = (unsigned short)floor((_values[i] - minTemp) / m_stepX);
			if (index < m_nbBins)
				m_bins[index]++;
		}
		m_maxY = 0.;
		for (int i = 0; i < m_nbBins; i++) {
			m_ts[i] = minTemp + (float)i * m_stepX + 0.5f * m_stepX;
			if (m_bins[i] > m_maxY)
				m_maxY = m_bins[i];
		}
	}

	const size_t Histogram::memorySize() const
	{
		size_t memoryS = 0;
		memoryS += 3 * sizeof(float*);
		if (!m_values.empty())
			memoryS += m_nbValues * sizeof(float);
		if (!m_bins.empty())
			memoryS += m_nbBins * sizeof(float);
		if (!m_ts.empty())
			memoryS += m_nbBins * sizeof(float);
		memoryS += 5 * sizeof(float);
		memoryS += 6 * sizeof(float);
		memoryS += 3 * sizeof(bool);
		return memoryS;
	}
}

