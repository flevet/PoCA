/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectColocalizationWidget.hpp
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

#ifndef ObjectColocalizationWidget_h__
#define ObjectColocalizationWidget_h__

#include <QtWidgets/QTabWidget>

#include <Plot/FilterHistogramWidget.hpp>
#include <DesignPatterns/Observer.hpp>
#include <General/Command.hpp>
#include <General/Palette.hpp>

class QPushButton;

//! [0]
class ObjectColocalizationWidget : public QTabWidget, public poca::core::ObserverForMediator {
	Q_OBJECT

public:
	ObjectColocalizationWidget(poca::core::MediatorWObjectFWidgetInterface*, QWidget* = 0);
	~ObjectColocalizationWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&);
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

protected:

protected slots:
	void actionNeeded();
	void actionNeeded(bool);

protected:
	QTabWidget* m_parentTab;
	poca::core::MediatorWObjectFWidgetInterface* m_mediator;

	QWidget* m_lutsWidget, * m_buttonsWidget, * m_delaunayTriangulationFilteringWidget, * m_emptyWidget;
	std::vector <std::pair<QPushButton*, std::string>> m_lutButtons;
	std::pair<QPushButton*, std::string> m_hilowButton;
	std::vector <poca::plot::FilterHistogramWidget*> m_histWidgets;
	QPushButton* m_displayButton, * m_fillButton, * m_pointRenderButton, * m_shapeRenderButton, * m_bboxSelectionButton, * m_exportButton, * m_selectionButton;

	QTableWidget* m_tableObjects, * m_tableInfos;


	poca::core::MyObjectInterface* m_object;
};

//! [0]
#endif

