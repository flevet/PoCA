/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CustomizedSlider.cpp
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

#include "CustomizedSlider.hpp"

namespace poca::plot {
	CustomizedSlider::CustomizedSlider(float _min, float _max, int nbSteps, QWidget* parent)
		: QWidget(parent),
		m_minVal(_min),
		m_maxVal(_max),
		m_sliderSteps(nbSteps)
	{
		// Slider
		m_slider = new QSlider(Qt::Horizontal);
		m_slider->setRange(0, m_sliderSteps);
		m_slider->setValue(0);  // initial position
		m_slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

		// Min / Max labels
		m_minLEdit = new QLineEdit(QString::number(m_minVal, 'f', 3));
		m_minLEdit->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
		m_maxLEdit = new QLineEdit(QString::number(m_maxVal, 'f', 3));
		m_maxLEdit->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

		// Layout: single row
		QHBoxLayout* hLayout = new QHBoxLayout;
		hLayout->addWidget(m_minLEdit);
		hLayout->addWidget(m_slider);
		hLayout->addWidget(m_maxLEdit);

		setLayout(hLayout);

		// Live tooltip during drag
		connect(m_slider, &QSlider::sliderMoved, this, &CustomizedSlider::showTooltip);

		// Emit signal when value changes (optional)
		connect(m_slider, &QSlider::valueChanged, this, &CustomizedSlider::emitValueChanged);
	
	
		connect(m_minLEdit, &QLineEdit::editingFinished, this, &CustomizedSlider::modifyMinMax);
		connect(m_maxLEdit, &QLineEdit::editingFinished, this, &CustomizedSlider::modifyMinMax);
	}

	float CustomizedSlider::value() const
	{
		return m_minVal + (m_maxVal - m_minVal) * (float(m_slider->value()) / m_sliderSteps);
	}

	void CustomizedSlider::setMaxValue(float _val)
	{
		m_maxVal = _val;
	}

	void CustomizedSlider::showTooltip(int value)
	{
		float alpha = m_minVal + (m_maxVal - m_minVal) * (float(value) / m_sliderSteps);

		// Get slider handle position using QStyle
		QStyleOptionSlider opt;
		opt.initFrom(m_slider);
		opt.orientation = m_slider->orientation();
		opt.minimum = m_slider->minimum();
		opt.maximum = m_slider->maximum();
		opt.sliderPosition = value;
		opt.sliderValue = value;

		QRect handleRect = m_slider->style()->subControlRect(
			QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, m_slider);

		QPoint handleCenter = m_slider->mapToGlobal(handleRect.center());
		QPoint tooltipPos(handleCenter.x(), handleCenter.y() - 30);  // offset above handle

		QToolTip::showText(tooltipPos, QString::number(alpha, 'f', 3), m_slider);
	}

	void CustomizedSlider::emitValueChanged(int value)
	{
		float alpha = m_minVal + (m_maxVal - m_minVal) * (float(value) / m_sliderSteps);
		emit changedValue(alpha);
	}

	void CustomizedSlider::modifyMinMax()
	{
		QObject* sender = QObject::sender();
		if (sender == m_minLEdit) {
			bool ok;
			float tmp = m_minLEdit->text().toFloat(&ok);
			if (ok)
				m_minVal = tmp;
		}
		else if (sender == m_maxLEdit) {
			bool ok;
			float tmp = m_maxLEdit->text().toFloat(&ok);
			if (ok)
				m_maxVal = tmp;
		}
	}
}