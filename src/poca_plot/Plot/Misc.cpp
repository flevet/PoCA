/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Misc.cpp
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

#include <QtGui/QPainter>

#include "Misc.h"

namespace poca::core {
	QImage generateImage(unsigned int _w, unsigned int _h, poca::core::Palette* _pal)
	{
		std::vector <float> pos;
		std::vector <poca::core::Color4uc> colors;
		_pal->getGradientInfos(pos, colors);
		QImage image(_w, _h, QImage::Format_ARGB32_Premultiplied);
		QPainter painter(&image);
		QLinearGradient gradient;
		for (size_t n = 0; n < pos.size(); n++)
			gradient.setColorAt(pos[n], QColor(colors[n][0], colors[n][1], colors[n][2]));
		gradient.setStart(0.0, 0.0);
		gradient.setFinalStop((qreal)_w, 0.0);
		painter.fillRect(0, 0, _w, _h, gradient);
		return image;
	}
}

