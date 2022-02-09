/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Histogram.hpp
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

#ifndef Histogram_h__
#define Histogram_h__

#include <vector>

#include "../Interfaces/HistogramInterface.hpp"
#include "ArrayStatistics.hpp"

namespace poca::core {

	class Histogram : public HistogramInterface {
	public:
		Histogram();
		Histogram(const std::vector <float>&, const std::size_t, const bool, const int = 100, const bool = false, const float = 0., const bool = false, const float = 0.);
		Histogram(const float*, const std::size_t, const bool, const int = 100, const bool = false, const float = 0., const bool = false, const float = 0.);
		Histogram(const Histogram&);
		~Histogram();

		Histogram& operator=(const Histogram&);

		void setHistogram(const float*, const std::size_t, const bool, const int = 100, const bool = false, const float = 0., const bool = false, const float = 0.);
		void changeHistogramBounds(const float, const float);
		void setNbBins(const std::size_t);
		void setNbBins(const std::size_t, const std::vector <float>&);
		void setNbBins(const std::size_t, const float*, const std::size_t);

		void eraseBounds();
		void resetBounds();

		const size_t memorySize() const;

		inline const float* getValuesPtr() const { return m_values.data(); }
		inline const float* getBinsPtr() const { return m_bins.data(); }
		inline const float* getTsPtr() const { return m_ts.data(); }

		const std::vector <float>& getValues() const { return m_values; }
		std::vector <float>& getValues() { return m_values; }
		const std::vector <float>& getBins() const { return m_bins; }
		std::vector <float>& getBins() { return m_bins; }
		const std::vector <float>& getTs() const { return m_ts; }
		std::vector <float>& getTs() { return m_ts; }

		const std::size_t getNbValues() const { return m_nbValues; }
		const std::size_t getNbBins() const { return m_nbBins; }
		const float getMean() const { return m_stats.getData(ArrayStatistics::Mean); }
		const float getMedian() const { return m_stats.getData(ArrayStatistics::Median); }
		const float getStdDev() const { return m_stats.getData(ArrayStatistics::StdDev); }
		const float getMin() const { return (m_isMinDefined) ? m_minDefined : m_stats.getData(ArrayStatistics::Min); }
		const float getMax() const { return (m_isMaxDefined) ? m_maxDefined : m_stats.getData(ArrayStatistics::Max); }
		const float getStepX() const { return m_stepX; }
		const float getMaxY() const { return m_maxY; }
		const float getCurrentMin() const { return m_currentMin; }
		void setCurrentMin(const float _val) { m_currentMin = _val; }
		const float getCurrentMax() const { return m_currentMax; }
		void setCurrentMax(const float _val) { m_currentMax = _val; }

		inline const ArrayStatistics& getStats() const { return m_stats; }
		inline void setLog(const bool _isLog) { m_isLog = _isLog; }
		inline const bool isLog() const { return m_isLog; }

		inline const bool isMinDefined() const { return m_isMinDefined; }
		inline const bool isMaxDefined() const { return m_isMaxDefined; }

	protected:
		std::vector<float> m_values, m_bins, m_ts;
		std::size_t m_nbValues, m_nbBins;
		ArrayStatistics m_stats;
		float m_stepX, m_maxY, m_currentMin, m_currentMax;
		bool m_isMinDefined, m_isMaxDefined, m_isLog;
		float m_minDefined, m_maxDefined;

		//EquationFit * m_eqn;
	};

}
#endif // Histogram_h__

