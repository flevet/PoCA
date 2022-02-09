/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DBSCANWidget.hpp
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
#include <General/Vec4.hpp>

class QPushButton;
class QCheckBox;
class QLabel;
class QTableWidget;

namespace poca::plot {
	class QCPHistogram;
}
//! [0]
class DBSCANWidget : public QWidget, public poca::core::ObserverForMediator {
	Q_OBJECT

public:
	DBSCANWidget(poca::core::MediatorWObjectFWidgetInterface*, QWidget* = 0);
	~DBSCANWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&);
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

protected:

protected slots:
	void actionNeeded();
	void actionNeeded(bool);
	void updateDBSCANResults();

protected:
	QTabWidget* m_parentTab;
	poca::core::MediatorWObjectFWidgetInterface* m_mediator;

	QLabel* m_distanceDBScanLbl;
	QLineEdit* m_leditDistanceDBScan, * m_leditMinDDBScan, * m_leditMinPtsPerCluster;
	QPushButton* m_buttonDBScan, * m_buttonExportDBSCANRes, * m_displayButton, * m_creationFlteredObjectsButton;
	poca::plot::QCPHistogram* m_customPlotDBSCAN;
	QTableWidget* m_tableObjs;
	poca::core::Color4D m_colorBack, m_colorObj;
	
	poca::core::MyObjectInterface* m_object;
};

//! [0]
#endif

