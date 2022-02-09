/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      KRipleyWidget.hpp
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

#ifndef KRipleyWidgetWidget_h__
#define KRipleyWidgetWidget_h__

#include <QtWidgets/QTabWidget>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLineEdit>

#include <DesignPatterns/Observer.hpp>
#include <General/Command.hpp>

class QPushButton;
class QCheckBox;
class QLabel;
class QCustomPlot;

//! [0]
class KRipleyWidget : public QWidget, public poca::core::ObserverForMediator {
	Q_OBJECT

public:
	KRipleyWidget(poca::core::MediatorWObjectFWidgetInterface*, QWidget* = 0);
	~KRipleyWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&);
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

	void setKripleyCurveDisplay();

protected:

protected slots:
	void actionNeeded();
	void toggleRipleyFunctionDisplay(bool);
	void exportKRipleyResults();

protected:
	QTabWidget* m_parentTab;
	poca::core::MediatorWObjectFWidgetInterface* m_mediator;

	QLabel* m_minKRipleyLbl, * m_maxKRipleyLbl, * m_stepKRipleyLbl, * m_resKRipleyLbl;
	QLineEdit* m_minKRipleyEdit, * m_maxKRipleyEdit, * m_stepKRipleyEdit;
	QCheckBox* m_cboxROIsKRipley, * m_cboxLsDisplayKRipley;
	QPushButton* m_buttonKRipley, * m_buttonExportKRipleyRes;
	QCustomPlot* m_customPlotKRipley;
	bool m_lsSelected;
	
	poca::core::MyObjectInterface* m_object;
};

//! [0]
#endif

