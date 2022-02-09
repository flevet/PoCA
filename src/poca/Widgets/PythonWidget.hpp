/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonWidget.hpp
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

#ifndef PythonWidget_h__
#define PythonWidget_h__

#ifndef NO_PYTHON

#include <QtWidgets/QWidget>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QPushButton>

#include "DesignPatterns/Observer.hpp"
#include "DesignPatterns/MediatorWObjectFWidget.hpp"
#include "General/Command.hpp"

class QPlainTextEdit;
class QDockWidget;
class QLabel;

class PythonWidget: public QWidget, public poca::core::ObserverForMediator{
	Q_OBJECT

public:
	PythonWidget(poca::core::MediatorWObjectFWidget *, QWidget* = 0, Qt::WindowFlags = 0 );
	~PythonWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo & );
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

protected slots:
	void actionNeeded();
	void actionNeeded( int );
	void actionNeeded( bool );

protected:
	void executeNena();
	void executeEllipsoidFit();
	void executeCAML();

protected:
	QTabWidget* m_parentTab;

	QGroupBox* m_groupPreloadedPythonFiles, * m_groupLoadPythonFiles;
	std::vector <std::pair <QPushButton*, std::string>> m_buttonsPreloaded;
	
	poca::core::MyObjectInterface* m_object;
	poca::core::MediatorWObjectFWidget * m_mediator;
};
#endif
#endif // PythonWidget_h__

