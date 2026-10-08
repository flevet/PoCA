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
#include <memory>

namespace poca::core {
	MyData::MyData() :m_histogram(nullptr), m_logHistogram(nullptr), m_log(false), m_computeLog(false)
	{
	}

	MyData::MyData(HistogramInterface* _hist, HistogramInterface* _logHist): m_histogram(nullptr), m_logHistogram(nullptr), m_log(false), m_computeLog(false)
	{
		m_histogram = _hist;
		m_logHistogram = _logHist;
	}

	MyData::MyData(HistogramInterface* _hist, const bool _computeLogHisto, const bool _deferLog): m_histogram(nullptr), m_logHistogram(nullptr), m_log(false), m_computeLog(_computeLogHisto)
	{
		m_histogram = _hist;
		if(m_computeLog && !_deferLog && !m_histogram->valuesUnloaded())
			m_logHistogram = m_histogram->computeLogHistogram();
	}

	MyData::MyData(const MyData& _o) : m_histogram(nullptr), m_logHistogram(nullptr), m_log(_o.m_log), m_computeLog(_o.m_computeLog), m_deferredLog(_o.m_deferredLog)
	{
		std::unique_ptr<HistogramInterface> original(_o.m_histogram ? _o.m_histogram->clone() : nullptr);
		std::unique_ptr<HistogramInterface> log(_o.m_logHistogram ? _o.m_logHistogram->clone() : nullptr);
		m_histogram = original.release();
		m_logHistogram = log.release();
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
		if (m_computeLog && !m_histogram->valuesUnloaded()) {
			std::unique_ptr<HistogramInterface> log(m_histogram->computeLogHistogram());
			delete m_logHistogram;
			m_logHistogram = log.release();
		}
	}

	const size_t MyData::nbElements() const
	{
		return m_histogram->nbElements();
	}

	void MyData::setLog(const bool _val) {
		m_deferredLog = false;
		if (_val == m_log) return;
		if (_val && !m_logHistogram) {
			if (!m_computeLog) throw std::invalid_argument("Feature does not support logarithmic display");
			m_logHistogram = m_histogram->computeLogHistogram();
		}
		HistogramInterface* current = m_log ? m_logHistogram : m_histogram;
		HistogramInterface* other = !m_log ? m_logHistogram : m_histogram;
		if (current == NULL) return;
		if (other && other->hasDisplayBounds() && current->hasDisplayBounds()) {
			const float minV = current->getCurrentMin(), maxV = current->getCurrentMax();
			const float low = static_cast<float>(m_log ? std::pow(10., minV) : std::log10(minV));
			const float high = static_cast<float>(m_log ? std::pow(10., maxV) : std::log10(maxV));
			// A linear interval crossing zero maps to the available positive logarithmic domain.
			other->setCurrentMin(std::isfinite(low) ? low : other->getMin());
			other->setCurrentMax(std::isfinite(high) ? high : other->getMax());
		}
		m_log = _val;
	}
}

