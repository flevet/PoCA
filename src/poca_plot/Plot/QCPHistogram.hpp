/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      QCPHistogram.hpp
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

#ifndef QCPHistogram_h__
#define QCPHistogram_h__

#include "qcustomplot.h"
#include "Interfaces/HistogramInterface.hpp"
#include "Interfaces/PaletteInterface.hpp"

namespace poca::plot {

	class QCPHistogram : public QCustomPlot
	{
		Q_OBJECT

	public:
		QCPHistogram(QWidget* = 0);
		~QCPHistogram();

		void mousePressEvent(QMouseEvent*);
		void mouseMoveEvent(QMouseEvent*);
		void mouseReleaseEvent(QMouseEvent*);

		virtual void update();

		QSize sizeHint() const;

		inline void setInfos(const QString& _n, poca::core::HistogramInterface* _h, poca::core::PaletteInterface* _p) { this->setToolTip(_n); setHistogram(_h); setPalette(_p); setName(_n); }
		inline void setHistogram(poca::core::HistogramInterface* _hist) { m_histogram = _hist; }
		inline void setPalette(poca::core::PaletteInterface* _palette) { m_palette = _palette; }
		inline void setName(const QString& _name) { m_name = _name; }
		inline poca::core::HistogramInterface* getHistogram() const { return m_histogram; }
		inline poca::core::PaletteInterface* getPalette() const { return m_palette; }
		inline const QString& getName() const { return m_name; }
		inline const float getCurrentMin() const { return m_currentMin; }
		inline const float getCurrentMax() const { return m_currentMax; }
		inline void setCurrentMin(const double _val) { m_currentMin = _val; }
		inline void setCurrentMax(const double _val) { m_currentMax = _val; }

	signals:
		void actionNeededSignal(const QString&);

	protected:
		bool m_buttonLeft, m_buttonRight, m_buttonMiddle;
		float m_currentMin, m_currentMax, m_currentX;

		poca::core::HistogramInterface* m_histogram;
		poca::core::PaletteInterface* m_palette;
		QString m_name;
	};
}

#endif

