/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      QCPGraphWithColor.cpp
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

#include "QCPGraphWithColor.hpp"

namespace poca::plot {

    QCPGraphWithColor::QCPGraphWithColor(QCPAxis* keyAxis, QCPAxis* valueAxis) :QCPGraph(keyAxis, valueAxis), m_palette(NULL)
    {

    }

    QCPGraphWithColor::~QCPGraphWithColor()
    {
        if (m_palette != NULL)
            delete m_palette;
    }

    void QCPGraphWithColor::setInfos(const QVector<double>& _colors, poca::core::PaletteInterface* _palette)
    {
        m_colors = _colors;
        if (m_palette != NULL)
            delete m_palette;
        m_palette = (_palette == NULL) ? _palette : _palette->copy();
    }

    void QCPGraphWithColor::drawFill(QCPPainter* painter, QVector<QPointF>* lines) const
    {
        if (m_palette == NULL)
            QCPGraph::drawFill(painter, lines);
        else {
            if (mLineStyle == lsImpulse) return; // fill doesn't make sense for impulse plot
            if (painter->brush().style() == Qt::NoBrush || painter->brush().color().alpha() == 0) return;

            applyFillAntialiasingHint(painter);
            QVector<QCPDataRange> segments = getNonNanSegments(lines, keyAxis()->orientation());
            if (!mChannelFillGraph)
            {
                // draw base fill under graph, fill goes all the way to the zero-value-line:
                for (int i = 0; i < segments.size(); ++i) {
                    QPolygonF result(segments[i].size() + 2);

                    QPointF lineBase = getFillBasePoint(lines->at(segments[i].begin()));
                    for (size_t n = 1; n < lines->size(); n++) {
                        QPointF p1 = lines->at(n - 1), p2 = lines->at(n);
                        QPolygonF poly;
                        poly.push_back(QPointF(p1.x(), lineBase.y()));
                        poly.push_back(QPointF(p1.x(), p1.y()));
                        poly.push_back(QPointF(p2.x(), p2.y()));
                        poly.push_back(QPointF(p2.x(), lineBase.y()));

                        poca::core::Color4uc color = m_palette->getColor(m_colors.at(n));
                        QColor c(color[0], color[1], color[2]);
                        painter->setBrush(c);
                        painter->drawPolygon(poly);
                    }
                }
            }
        }
    }
}

