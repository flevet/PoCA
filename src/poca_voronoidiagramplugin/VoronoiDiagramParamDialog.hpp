/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramParamDialog.hpp
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

#ifndef VoronoiDiagramParamDialog_h__
#define VoronoiDiagramParamDialog_h__

#include <QtWidgets/QDialog>
#include <QtWidgets/QCheckBox>
#include <vector>

class QLineEdit;

class VoronoiDiagramParamDialog : public QDialog{
	Q_OBJECT

public:
	VoronoiDiagramParamDialog(const bool, const bool, const bool, QWidget * = 0, Qt::WindowFlags = 0);
	~VoronoiDiagramParamDialog();

	inline const bool isCreateCells() const { return m_cbox->isChecked(); }
	inline const bool isTabShown() const { return m_cboxDisplay->isChecked(); }
	inline const bool isVoronoiOnSphere() const { return m_cboxVoronoiOnSphere->isChecked(); }

protected slots:
	

protected:
	QCheckBox* m_cbox, * m_cboxDisplay, * m_cboxVoronoiOnSphere;
};

#endif // ColocalizationChoiceDialog_h__

