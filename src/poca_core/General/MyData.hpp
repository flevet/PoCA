/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyData.hpp
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

#ifndef MyData_h__
#define MyData_h__

#include <vector>
#include <algorithm>
#include <execution>

#include "Histogram.hpp" 

namespace poca::core {

	class MyData {
	public:
		MyData();
		MyData(HistogramInterface*, HistogramInterface* = NULL);
		MyData(HistogramInterface*,const bool = false);
		MyData(const MyData&);
		~MyData();

		void finalizeData();

		template <class T>
		const std::vector < T >& getOriginalData() const;
		template <class T>
		std::vector < T >& getOriginalData();
		template <class T>
		const std::vector < T >& getData() const;
		template <class T>
		std::vector < T >& getData();
		const size_t nbElements() const;
		void setLog(const bool);

		inline HistogramInterface* getHistogram() { return m_log ? m_logHistogram : m_histogram; }
		inline HistogramInterface* getHistogram() const { return m_log ? m_logHistogram : m_histogram; }
		inline HistogramInterface* getOriginalHistogram() { return m_histogram; }
		inline HistogramInterface* getOriginalHistogram() const { return m_histogram; }

		inline const bool isLog() const { return m_log; }

	protected:
		HistogramInterface* m_histogram, * m_logHistogram;
		bool m_log{ false }, m_computeLog{ true };
	};

	

	/*template <class T>
	MyData::MyData(const std::vector < T >& _data, const bool _computeLog) : m_histogram(nullptr), m_logHistogram(nullptr), m_log(false)
	{
		m_histogram = new Histogram<T>(_data, _data.size(), false);

		if (_computeLog)
			m_logHistogram = m_histogram->computeLogHistogram();
	}

	template <class T>
	MyData::MyData(const MyData& _o) : m_log(_o.m_log)
	{
		m_histogram = new Histogram<T>(*_o.m_histogram);
		if (_o.m_logHistogram != nullptr)
			m_logHistogram = new Histogram(*_o.m_logHistogram);
	}*/

	template <class T>
	const std::vector < T >& MyData::getOriginalData() const
	{
		return dynamic_cast<Histogram<T>*>(m_histogram)->getValues();
	}

	template <class T>
	std::vector < T >& MyData::getOriginalData()
	{
		return dynamic_cast<Histogram<T>*>(m_histogram)->getValues();
	}

	template <class T>
	const std::vector < T >& MyData::getData() const
	{
		return m_log ? dynamic_cast<Histogram<T>*>(m_logHistogram)->getValues() : dynamic_cast<Histogram<T>*>(m_histogram)->getValues();
	}

	template <class T>
	std::vector < T >& MyData::getData()
	{
		return m_log ? dynamic_cast<Histogram<T>*>(m_logHistogram)->getValues() : dynamic_cast<Histogram<T>*>(m_histogram)->getValues();
	}
}

#endif

