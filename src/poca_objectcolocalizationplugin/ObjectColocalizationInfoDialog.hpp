/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectColocalizationInfoDialog.hpp
*
* Copyright: Florian Levet (2020-2021)
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

#ifndef ObjectColocalizationInfoDialog_h__
#define ObjectColocalizationInfoDialog_h__

#include <QtWidgets/QDialog>
#include <QtWidgets/QCheckBox>
#include <vector>

class QLineEdit;

class ObjectColocalizationInfoDialog : public QDialog{
	Q_OBJECT

public:
	ObjectColocalizationInfoDialog(const bool, const float, const uint32_t, const uint32_t, QWidget * = 0, Qt::WindowFlags = 0);
	~ObjectColocalizationInfoDialog();

	const float getDistance() const;
	const uint32_t getNbSubdivision() const;
	const uint32_t getMinNbPoints() const;

	inline const bool isSamplingEnabled() const { return m_cbox->isChecked(); }

protected slots:
	

protected:
	QCheckBox* m_cbox;
	QLineEdit* m_ledit2D, * m_ledit3D, * m_leditMinLocs;
};

#endif // ColocalizationChoiceDialog_h__

