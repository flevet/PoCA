/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyData.cpp
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

#include "MyData.hpp"

namespace poca::core {
	MyData::MyData() :m_histogram(nullptr), m_logHistogram(nullptr), m_log(false), m_computeLog(false)
	{
	}

	MyData::MyData(HistogramInterface* _hist, HistogramInterface* _logHist): m_histogram(nullptr), m_logHistogram(nullptr), m_log(false), m_computeLog(false)
	{
		m_histogram = _hist;
		m_logHistogram = _logHist;
	}

	MyData::MyData(HistogramInterface* _hist, const bool _computeLogHisto): m_histogram(nullptr), m_logHistogram(nullptr), m_log(false), m_computeLog(_computeLogHisto)
	{
		m_histogram = _hist;
		if(m_computeLog)
			m_logHistogram = m_histogram->computeLogHistogram();
	}

	MyData::MyData(const MyData& _o)
	{
		m_histogram = _o.m_histogram;
		m_logHistogram = _o.m_logHistogram;
	}

	MyData::~MyData()
	{
		if (m_histogram != nullptr)
			delete m_histogram;
		if (m_logHistogram != nullptr)
			delete m_logHistogram;
		m_histogram = m_logHistogram = nullptr;
	}

	void MyData::finalizeData()
	{
		m_histogram->setHistogram(false);
		if (m_computeLog) 
			m_logHistogram = m_histogram->computeLogHistogram();
	}

	const size_t MyData::nbElements() const
	{
		return m_histogram->nbElements();
	}

	void MyData::setLog(const bool _val) {
		HistogramInterface* current = m_log ? m_logHistogram : m_histogram;
		HistogramInterface* other = !m_log ? m_logHistogram : m_histogram;
		float minV = current->getCurrentMin(), maxV = current->getCurrentMax();
		float modified_min = (float)(m_log ? pow(10, minV) : log10(minV));
		float modified_max = (float)(m_log ? pow(10, maxV) : log10(maxV));
		other->setCurrentMin(modified_min);
		other->setCurrentMax(modified_max);
		m_log = _val;
	}
}

