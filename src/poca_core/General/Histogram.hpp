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
#include <float.h>
#include <iostream>
#include <ctime>
#include <execution>
#include <fstream>

#include "../Interfaces/HistogramInterface.hpp"
#include "ArrayStatistics.hpp"
#include "../Cuda/CoreMisc.h"

namespace poca::core {

	template <class T>
	class Histogram : public HistogramInterface {
	public:
		Histogram();
		Histogram(const std::vector <T>&, const bool, const int = 100, const bool = false, const float = 0., const bool = false, const float = 0.);
		//Histogram(const float*, const std::size_t, const bool, const int = 100, const bool = false, const float = 0., const bool = false, const float = 0.);
		Histogram(const Histogram<T>&);
		~Histogram();

		Histogram& operator=(const Histogram<T>&);

		void setHistogram(const std::vector<T>&, const bool, const int = 100, const bool = false, const float = 0., const bool = false, const float = 0.);
		void setHistogram(const bool, const int = 100, const bool = false, const float = 0., const bool = false, const float = 0.);
		void changeHistogramBounds(const float, const float);
		void setNbBins(const std::size_t);
		void setNbBins(const std::size_t, const std::vector <T>&);
		//void setNbBins(const std::size_t, const float*, const std::size_t);

		void eraseBounds();
		void resetBounds();

		const size_t memorySize() const;

		inline const T* getValuesPtr() const { return m_values.data(); }
		inline const float* getBinsPtr() const { return m_bins.data(); }
		inline const float* getTsPtr() const { return m_ts.data(); }

		const std::vector <T>& getValues() const { return m_values; }
		std::vector <T>& getValues() { return m_values; }
		const std::vector <float>& getBins() const { return m_bins; }
		std::vector <float>& getBins() { return m_bins; }
		const std::vector <float>& getTs() const { return m_ts; }
		std::vector <float>& getTs() { return m_ts; }

		void setSelection(std::vector <bool>&);
		void saveValues(std::ofstream&) const;
		HistogramInterface* computeLogHistogram() const;
		const size_t nbElements() const;

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

		void setInteraction(const bool _val) { m_hasInteraction = _val; }
		virtual bool hasInteraction() const { return m_hasInteraction; }

	protected:
		std::vector<T> m_values;
		std::vector<float> m_bins, m_ts;
		std::size_t m_nbValues{ 0 }, m_nbBins{ 0 };
		ArrayStatistics m_stats;
		float m_stepX{ 0.f }, m_maxY{ FLT_MAX }, m_currentMin{ -FLT_MAX }, m_currentMax{ FLT_MAX };
		bool m_isMinDefined{ false }, m_isMaxDefined{ false }, m_isLog{ false }, m_hasLogHistogram{ true }, m_hasInteraction{ true };
		float m_minDefined{ 0.f }, m_maxDefined{ 0.f };
		//EquationFit * m_eqn;
	};

	template <class T>
	Histogram<T>::Histogram()
	{
	}

	template <class T>
	Histogram<T>::Histogram(const std::vector <T>& _vals, const bool _isLog, const int _nbBins, const bool _isMinDefined, const float _minDefined, const bool _isMaxDefined, const float _maxDefined)
	{
		setHistogram(_vals, _isLog, _nbBins, _isMinDefined, _minDefined, _isMaxDefined, _maxDefined);
	}

	/*Histogram<T>::Histogram(const float* _vals, const std::size_t _nbVals, const bool _isLog, const int _nbBins, const bool _isMinDefined, const float _minDefined, const bool _isMaxDefined, const float _maxDefined)
	{
		setHistogram(_vals, _nbVals, _isLog, _nbBins, _isMinDefined, _minDefined, _isMaxDefined, _maxDefined);
	}*/

	template <class T>
	Histogram<T>::Histogram(const Histogram& _o) :m_values(_o.m_values), m_bins(_o.m_bins), m_ts(_o.m_ts), m_nbValues(_o.m_nbValues), m_nbBins(_o.m_nbBins),
		m_stats(_o.m_stats), m_stepX(_o.m_stepX), m_maxY(_o.m_maxY), m_currentMin(_o.m_currentMin),
		m_currentMax(_o.m_currentMax), m_isMinDefined(_o.m_isMinDefined), m_isMaxDefined(_o.m_isMaxDefined), m_isLog(_o.m_isLog),
		m_minDefined(_o.m_minDefined), m_maxDefined(_o.m_maxDefined)
	{
	}

	template <class T>
	Histogram<T>& Histogram<T>::operator=(const Histogram<T>& _o)
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

	template <class T>
	Histogram<T>::~Histogram()
	{
	}

	template <class T>
	void Histogram<T>::setHistogram(const std::vector<T>& _vals, const bool _isLog, const int _nbBins, const bool _isMinDefined, const float _minDefined, const bool _isMaxDefined, const float _maxDefined)
	{
		clock_t t1 = clock(), t2;
		m_values.clear();
		std::copy(_vals.begin(), _vals.end(), std::back_inserter(m_values));
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		//std::cout << "Time copy " << elapsed << std::endl;

		setHistogram(_isLog, _nbBins, _isMinDefined, _minDefined, _isMaxDefined, _maxDefined);
	}

	template <class T>
	void Histogram<T>::setHistogram(const bool _isLog, const int _nbBins, const bool _isMinDefined, const float _minDefined, const bool _isMaxDefined, const float _maxDefined)
	{
		m_isMinDefined = _isMinDefined; m_isMaxDefined = _isMaxDefined; m_minDefined = _minDefined; m_maxDefined = _maxDefined; m_isLog = _isLog;

		m_nbValues = m_values.size();
		clock_t t1 = clock(), t2;

		m_stats = ArrayStatistics::generateArrayStatistics(m_values, m_nbValues);
		m_currentMin = m_stats.getData(ArrayStatistics::Min);
		m_currentMax = m_stats.getData(ArrayStatistics::Max);
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		//std::cout << "Time for array statistics " << elapsed << std::endl;

		setNbBins(_nbBins);
	}

	template <class T>
	void Histogram<T>::changeHistogramBounds(const float _min, const float _max)
	{
		m_isMinDefined = _min != FLT_MAX;
		m_isMaxDefined = _min != FLT_MAX;
		if (m_minDefined || m_isMaxDefined) {
			m_minDefined = m_isMinDefined ? _min : m_minDefined;
			m_maxDefined = m_isMaxDefined ? _max : m_maxDefined;
			setNbBins(m_nbBins);
		}
	}

	template <class T>
	void Histogram<T>::eraseBounds()
	{
		m_currentMin = FLT_MIN;
		m_currentMax = FLT_MAX;
	}


	template <class T>
	void Histogram<T>::resetBounds()
	{
		m_currentMin = m_stats.getData(ArrayStatistics::Min);
		m_currentMax = m_stats.getData(ArrayStatistics::Max);
	}

	template <class T>
	void Histogram<T>::setNbBins(const std::size_t _nbBins)
	{
		setNbBins(_nbBins, m_values);
	}

	template <class T>
	void Histogram<T>::setNbBins(const std::size_t _nbBins, const std::vector <T>& _values)
	{
		//setNbBins(_nbBins, m_values.data(), m_values.size());

		clock_t t1 = clock(), t2;
		m_nbBins = _nbBins;

		m_bins.resize(m_nbBins, 0.);
		m_ts.resize(m_nbBins);

		float minTemp = (m_isMinDefined) ? m_minDefined : m_stats.getData(ArrayStatistics::Min);
		float maxTemp = (m_isMaxDefined) ? m_maxDefined : m_stats.getData(ArrayStatistics::Max);
		m_stepX = (maxTemp - minTemp) / (float)(m_nbBins - 1);

		computeHistogram(_values, m_bins, minTemp, maxTemp);

		/*for (unsigned int i = 0; i < _nbValues; i++) {
			if (_values[i] == -1) continue;
			unsigned short index = (unsigned short)floor((_values[i] - minTemp) / m_stepX);
			if (index < m_nbBins)
				m_bins[index]++;
		}*/
		m_maxY = 0.;
		for (int i = 0; i < m_nbBins; i++) {
			m_ts[i] = minTemp + (float)i * m_stepX + 0.5f * m_stepX;
			if (m_bins[i] > m_maxY)
				m_maxY = m_bins[i];
		}
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		//std::cout << "Time for histogram " << elapsed << std::endl;
	}

	/*void Histogram<T>::setNbBins(const std::size_t _nbBins, const float* _values, const std::size_t _nbValues)
	{
		clock_t t1 = clock(), t2;
		m_nbBins = _nbBins;

		m_bins.resize(m_nbBins, 0.);
		m_ts.resize(m_nbBins);

		float minTemp = (m_isMinDefined) ? m_minDefined : m_stats.getData(ArrayStatistics::Min);
		float maxTemp = (m_isMaxDefined) ? m_maxDefined : m_stats.getData(ArrayStatistics::Max);

		computeHistogram(_val)

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
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		//std::cout << "Time for histogram " << elapsed << std::endl;
	}*/

	template <class T>
	const size_t Histogram<T>::memorySize() const
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

	template <class T>
	void Histogram<T>::setSelection(std::vector <bool>& _selection)
	{
		if (_selection.size() != m_values.size()) return;
		T minV = T(getCurrentMin()), maxV = T(getCurrentMax());
#pragma omp parallel for
		for (int n = 0; n < m_values.size(); n++)
			_selection[n] = _selection[n] && minV <= m_values[n] && m_values[n] <= maxV;
	}

	template <class T>
	void Histogram<T>::saveValues(std::ofstream& _fs) const
	{
		for (size_t n = 0; n < m_values.size(); n++)
			_fs << m_values[n] << std::endl;
	}

	template <class T>
	HistogramInterface* Histogram<T>::computeLogHistogram() const
	{
		poca::core::Histogram<T>* logHistogram = new Histogram<T>();
		std::vector<T>& values = logHistogram->getValues();
		values.resize(m_values.size());
		std::transform(std::execution::par, m_values.begin(), m_values.end(), values.begin(), [](auto i) { return  (T)log10(i); });
		logHistogram->setHistogram(true);
		return logHistogram;
	}

	template <class T>
	const size_t Histogram<T>::nbElements() const
	{
		return m_values.size();
	}
}
#endif // Histogram_h__

