/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      QCPHistogram.cpp
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

#include "QCPHistogram.hpp"
#include "QCPGraphWithColor.hpp"

namespace poca::plot {

	QCPHistogram::QCPHistogram(QWidget* _parent) :QCustomPlot(_parent), m_buttonLeft(false), m_buttonRight(false), m_buttonMiddle(false), m_histogram(NULL)
	{
	}

	QCPHistogram::~QCPHistogram()
	{
	}

	void QCPHistogram::mousePressEvent(QMouseEvent* _event)
	{
		if (_event->modifiers() == Qt::ShiftModifier) {
			QCustomPlot::mousePressEvent(_event);
			return;
		}

		switch (_event->button())
		{
		case Qt::LeftButton:
		{
			QCPItemLine* lineC = qobject_cast<QCPItemLine*>(this->item(0));
			QCPItemLine* lineO = qobject_cast<QCPItemLine*>(this->item(1));
			if (!lineC || !lineO) return;

			m_buttonLeft = true;
			double x = this->xAxis->pixelToCoord(_event->pos().x());
			if (x > lineO->start->coords().x())
				x = lineO->start->coords().x();
			lineC->start->setCoords(x, lineC->start->coords().y());
			lineC->end->setCoords(x, lineC->end->coords().y());

			if (x < m_histogram->getMin())
				m_currentMin = m_histogram->getMin();
			else
				m_currentMin = x;

			this->replot();

			emit(actionNeededSignal("changeBoundsCustom"));
			break;
		}
		case Qt::MidButton:
		{
			m_buttonMiddle = true;
			m_currentX = this->xAxis->pixelToCoord(_event->pos().x());
			break;
		}
		case Qt::RightButton:
		{
			QCPItemLine* lineC = qobject_cast<QCPItemLine*>(this->item(1));
			QCPItemLine* lineO = qobject_cast<QCPItemLine*>(this->item(0));
			if (!lineC || !lineO) return;

			m_buttonRight = true;
			double x = this->xAxis->pixelToCoord(_event->pos().x());
			if (x < lineO->start->coords().x())
				x = lineO->start->coords().x();
			lineC->start->setCoords(x, lineC->start->coords().y());
			lineC->end->setCoords(x, lineC->end->coords().y());

			if (x > m_histogram->getMax())
				m_currentMax = m_histogram->getMax();
			else
				m_currentMax = x;

			this->replot();

			emit(actionNeededSignal("changeBoundsCustom"));
			break;
		}
		default:
			break;
		}
	}

	void QCPHistogram::mouseMoveEvent(QMouseEvent* _event)
	{
		if (_event->modifiers() == Qt::ShiftModifier) {
			QCustomPlot::mouseMoveEvent(_event);
			return;
		}

		QObject* sender = QObject::sender();

		double x = this->xAxis->pixelToCoord(_event->pos().x());
		double y = this->yAxis->pixelToCoord(_event->pos().y());

		if (m_buttonLeft) {
			QCPItemLine* lineC = qobject_cast<QCPItemLine*>(this->item(0));
			QCPItemLine* lineO = qobject_cast<QCPItemLine*>(this->item(1));
			if (lineC != NULL && lineO != NULL) {
				if (x > lineO->start->coords().x())
					x = lineO->start->coords().x();
				lineC->start->setCoords(x, lineC->start->coords().y());
				lineC->end->setCoords(x, lineC->end->coords().y());
				if (x < m_histogram->getMin())
					m_currentMin = m_histogram->getMin();
				else
					m_currentMin = x;

				float minH = m_histogram->getMin(), maxH = m_histogram->getMax();
				float inter = maxH - minH, minC = (m_currentMin - minH) / inter, maxC = (m_currentMax - minH) / inter;
				if(m_palette != NULL)
					m_palette->setFilterMinMax(minC, maxC);
				emit(actionNeededSignal("changeBoundsCustom"));
			}
			//}
		}

		if (m_buttonMiddle) {
			QCPItemLine* lineC = qobject_cast<QCPItemLine*>(this->item(0));
			QCPItemLine* lineO = qobject_cast<QCPItemLine*>(this->item(1));
			if (lineC != NULL && lineO != NULL) {
				float dx = x - m_currentX, xc = lineC->start->coords().x() + dx, xo = lineO->start->coords().x() + dx;
				m_currentX = x;
				if (xc < m_histogram->getMin() || xo > m_histogram->getMax())
					return;
				lineC->start->setCoords(xc, lineC->start->coords().y());
				lineC->end->setCoords(xc, lineC->end->coords().y());
				m_currentMin = xc;
				lineO->start->setCoords(xo, lineO->start->coords().y());
				lineO->end->setCoords(xo, lineO->end->coords().y());
				m_currentMax = xo;

				float minH = m_histogram->getMin(), maxH = m_histogram->getMax();
				float inter = maxH - minH, minC = (m_currentMin - minH) / inter, maxC = (m_currentMax - minH) / inter;
				if (m_palette != NULL)
					m_palette->setFilterMinMax(minC, maxC);
				emit(actionNeededSignal("changeBoundsCustom"));
			}
		}

		if (m_buttonRight) {
			QCPItemLine* lineC = qobject_cast<QCPItemLine*>(this->item(1));
			QCPItemLine* lineO = qobject_cast<QCPItemLine*>(this->item(0));
			if (lineC != NULL && lineO != NULL) {
				if (x < lineO->start->coords().x())
					x = lineO->start->coords().x();
				lineC->start->setCoords(x, lineC->start->coords().y());
				lineC->end->setCoords(x, lineC->end->coords().y());
				if (x > m_histogram->getMax())
					m_currentMax = m_histogram->getMax();
				else
					m_currentMax = x;

				float minH = m_histogram->getMin(), maxH = m_histogram->getMax();
				float inter = maxH - minH, minC = (m_currentMin - minH) / inter, maxC = (m_currentMax - minH) / inter;
				if (m_palette != NULL)
					m_palette->setFilterMinMax(minC, maxC);
				emit(actionNeededSignal("changeBoundsCustom"));
			}
		}
	}

	void QCPHistogram::mouseReleaseEvent(QMouseEvent* _event)
	{
		QCustomPlot::mouseReleaseEvent(_event);
		m_buttonLeft = m_buttonRight = m_buttonMiddle = false;
	}

	void QCPHistogram::update()
	{
		if (m_histogram == NULL) return;

		const std::vector <float>& ts = m_histogram->getTs();
		const std::vector <float>& bins = m_histogram->getBins();

		float maxY = 0.;
		unsigned int nbBins = bins.size();
		QVector<double> x1(nbBins), y1(nbBins);
		QVector<QCPGraphData> data(nbBins);
		for (unsigned int n = 0; n < nbBins; n++) {
			x1[n] = ts[n];
			y1[n] = bins[n];
			if (y1[n] > maxY) maxY = y1[n];

			data[n].key = x1[n];
			data[n].value = y1[n];
		}
		double minX = x1[0], maxX = x1[nbBins - 1], bin = (maxX - minX) / (double)nbBins, step = 1. / (double)nbBins;
		QVector<double> colors(nbBins);
		for (unsigned int n = 0; n < nbBins; n++)
			colors[n] = (double)n * step;

		m_currentMin = m_histogram->getCurrentMin();
		m_currentMax = m_histogram->getCurrentMax();

		this->clearGraphs();
		this->clearPlottables();
		this->clearItems();
		this->xAxis->grid()->setVisible(false);
		this->yAxis->grid()->setVisible(false);

		QCPLayoutGrid* layout = this->plotLayout();
		layout->setAutoMargins(QCP::msNone);
		layout->setMargins(QMargins(0, 0, 0, 0));

		// set axis ranges to show all data:
		this->xAxis->setRange(x1[0], x1[nbBins - 1]);
		this->yAxis->setRange(0, maxY);

		this->xAxis->setTicks(true);
		this->yAxis->setTicks(true);
		this->yAxis->setSubTicks(false);
		this->xAxis->setTickLabels(true);
		this->yAxis->setTickLabels(true);

		QSharedPointer<QCPAxisTickerFixed> fixedTicker(new QCPAxisTickerFixed);
		this->yAxis->setTicker(fixedTicker);

		fixedTicker->setTickStep(maxY); // tick step shall be 1.0
		fixedTicker->setScaleStrategy(QCPAxisTickerFixed::ssNone); // and no scaling of the tickstep (like multiples or powers) is allowed

		if (m_palette != NULL) {
			float minH = m_histogram->getMin(), maxH = m_histogram->getMax();
			float inter = maxH - minH, minC = (m_currentMin - minH) / inter, maxC = (m_currentMax - minH) / inter;
			m_palette->setFilterMinMax(minC, maxC);
		}
		
		QCPGraphWithColor* gwc = new QCPGraphWithColor(this->xAxis, this->yAxis);
		gwc->setInfos(colors, m_palette);
		QColor color(0, 0, 0);
		gwc->setLineStyle(QCPGraph::lsLine);
		gwc->setPen(QPen(color.lighter(200)));
		gwc->setBrush(QBrush(color));
		gwc->data()->set(data);

		this->setBackground(QColor(249, 249, 249));

		// Bounds min/max
		QCPItemLine* arrow = new QCPItemLine(this);
		QPen penLines(Qt::black);
		penLines.setWidth(1);
		arrow->setPen(penLines);
		arrow->start->setCoords(m_currentMin, 0.);
		arrow->end->setCoords(m_currentMin, maxY);
		arrow = new QCPItemLine(this);
		arrow->setPen(penLines);
		arrow->start->setCoords(m_currentMax, 0.);
		arrow->end->setCoords(m_currentMax, maxY);

		// add the text label at the top:
		//int w = this->width();
		QCPItemText* nameHisto = new QCPItemText(this);
		nameHisto->position->setType(QCPItemPosition::ptViewportRatio);
		nameHisto->position->setCoords(0.9, 0.2); // move 10 pixels to the top from bracket center anchor
		nameHisto->setPositionAlignment(Qt::AlignTop | Qt::AlignRight);
		nameHisto->setColor(Qt::black);
		nameHisto->setText(m_name);
		nameHisto->setFont(QFont(font().family(), 10, QFont::Bold));

		this->legend->clearItems();
		this->replot();
	}

	QSize QCPHistogram::sizeHint() const
	{
		return QSize(400, 30);
	}
}

