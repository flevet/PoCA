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
#include <QtWidgets/QListWidget>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QGridLayout>

#include <DesignPatterns/Observer.hpp>
#include <DesignPatterns/MediatorWObjectFWidget.hpp>
#include <General/Command.hpp>
#include <General/BasicComponent.hpp>

class QPlainTextEdit;
class QDockWidget;
class QLabel;

class PythonWidget: public QTabWidget, public poca::core::ObserverForMediator{
	Q_OBJECT

public:
	PythonWidget(poca::core::MediatorWObjectFWidget *, QWidget* = 0);
	~PythonWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo & );
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

	void execute(poca::core::CommandInfo*);
	void loadParameters(const nlohmann::json&);
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);

signals:
	void runMacro(std::vector<nlohmann::json>);

protected slots:
	void actionNeeded();
	void actionNeeded( int );
	void actionNeeded( bool );
	void updateChosenFunctionName(int);

protected:
	void populateListWidget(poca::core::BasicComponent*, QListWidget*);
	void populatePredefinedButtons();
	void addPredefinedButton(uint32_t);

protected:
	void executeNena();

	void executePythonScriptDisplayReturn(const poca::core::CommandInfo&);
	void executePythonScriptAddFeatureToComponent(const poca::core::CommandInfo&);

	QStringList identifyPythonFunctionNames(const QString&) const;

protected:
	QTabWidget* m_parentTab;

	QPushButton* m_buttonOpenFile, * m_buttonExecuteScript;
	QLabel* m_labelPythonFile;
	QLineEdit* m_editNameFunction;
	
	std::vector <QPushButton*> m_buttonsPreloaded, m_buttonsRemovePreloaded;
	QButtonGroup* m_bgroupGrid;

	QComboBox* m_BCCombo, * m_functionNameCombo;
	QListWidget* m_lists[2];
	QCheckBox* m_singleValCBox, * m_addFeatureCBox, * m_createNewDatasetCBox, * m_addToPredefinedModules;
	QLineEdit* m_nameFeatureEdit, * m_nameNewDatasetEdit, * m_namePredefinedCommand;
	
	poca::core::MyObjectInterface* m_object;
	poca::core::MediatorWObjectFWidget * m_mediator;

	nlohmann::json m_json;
	std::vector <poca::core::CommandInfo> m_pythonCommands;
	QGridLayout* m_layoutPredefined;
	int m_curRow = 0, m_curColumn = 0;
};
#endif
#endif // PythonWidget_h__

