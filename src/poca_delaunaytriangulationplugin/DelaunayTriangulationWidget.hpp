/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulationWidget.hpp
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

#ifndef DelaunayTriangulationWidget_h__
#define DelaunayTriangulationWidget_h__

#include <QtWidgets/QTabWidget>

#include <Plot/FilterHistogramWidget.hpp>
#include <DesignPatterns/Observer.hpp>
#include <General/Command.hpp>
#include <General/Palette.hpp>
#include <Interfaces/MyObjectInterface.hpp>

class QPushButton;

//! [0]
class DelaunayTriangulationWidget : public QWidget, public poca::core::ObserverForMediator {
	Q_OBJECT

public:
	DelaunayTriangulationWidget(poca::core::MediatorWObjectFWidgetInterface*, QWidget* = 0);
	~DelaunayTriangulationWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&);
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

protected:
	const float getCutDistance(bool*) const;
	const float getMinNbLocs(bool*) const;
	const float getMaxNbLocs(bool*) const;
	const float getMinArea(bool*) const;
	const float getMaxArea(bool*) const;

protected slots:
	void actionNeeded();
	void actionNeeded(bool);

protected:
	QTabWidget* m_parentTab;
	poca::core::MediatorWObjectFWidgetInterface* m_mediator;

	QWidget* m_lutsWidget, * m_delaunayTriangulationFilteringWidget, * m_emptyWidget;
	std::vector <std::pair<QPushButton*, std::string>> m_lutButtons;
	std::vector <poca::plot::FilterHistogramWidget*> m_histWidgets;
	QPushButton* m_displayButton, * m_fillButton, * m_creationFlteredObjectsButton, * m_bboxSelectionButton, * m_applyCutDButton, * m_invertSelectionButton;
	QLineEdit* m_cutDistanceEdit, * m_minLocLEdit, * m_maxLocLEdit, * m_minAreaLEdit, * m_maxAreaLEdit;
	QCheckBox* m_cboxApplyCutDistance, * m_cboxApplyMinLocs, * m_cboxApplyMaxLocs, * m_cboxApplyMinArea, * m_cboxApplyMaxArea, * m_cboxInROIs;

	poca::core::MyObjectInterface* m_object;
};

//! [0]
#endif

