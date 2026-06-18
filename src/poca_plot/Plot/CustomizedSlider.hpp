/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CustomizedSlider.hpp
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

#ifndef CustomizedSlider_h__
#define CustomizedSlider_h__

#include <QtWidgets/QSpinBox>
#include <QtWidgets/QSlider>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QStyleOptionSlider>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QToolTip>

namespace poca::plot {
	class CustomizedSlider : public QWidget
	{
		Q_OBJECT

	public:
		explicit CustomizedSlider(float _min, float _max, int nbSteps, bool = true, QWidget* parent = nullptr);

		float value() const;
		void setValue(float, bool = false);
		void setMaxValue(float);

		void modifyMinMax();

		void changeMin(float _val) { m_minLEdit->setText(QString::number(_val)); m_minVal = _val; }
		void changeMax(float _val) { m_maxLEdit->setText(QString::number(_val)); m_maxVal = _val; }
		void changeMinAndMax(float _valMin, float _valMax) { m_minLEdit->setText(QString::number(_valMin)); m_maxLEdit->setText(QString::number(_valMax)); m_minVal = _valMin; m_maxVal = _valMax; }

		float getValue() const { return m_value; }

	signals:
		void changedValue(float alpha);

	private:
		QSlider* m_slider;
		QLineEdit* m_minLEdit, * m_maxLEdit;
		float m_minVal, m_maxVal, m_value;
		int m_sliderSteps;

		void showTooltip(int value);

		void emitValueChanged(int value);
	};
}

#endif
