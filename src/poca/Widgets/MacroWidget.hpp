/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MacroWidget.hpp
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

#ifndef MacroWidget_h__
#define MacroWidget_h__

#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextEdit>

#include <DesignPatterns/Observer.hpp>
#include <DesignPatterns/MediatorWObjectFWidget.hpp>
#include <General/Command.hpp>
#include <General/json.hpp>

class QPlainTextEdit;
class QDockWidget;
class QLabel;
class QPushButton;

class MacroWidget: public QTabWidget, public poca::core::ObserverForMediator{
	Q_OBJECT

public:
	MacroWidget(poca::core::MediatorWObjectFWidget *, QWidget* = 0);
	~MacroWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo & );
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void execute(poca::core::CommandInfo*);
	void loadParameters(const nlohmann::json&);

	void getJsonsFromQString(const QString&, std::vector <nlohmann::json>&);
	void getJsonsFromTextEdit(QTextEdit*, std::vector <nlohmann::json>&);

	QTextEdit* getTextEdit() { return m_recordEdit; }
	std::vector <nlohmann::json>* getJson() { return &m_jsonRecord; }

signals:
	void runMacro(std::vector<nlohmann::json>);
	void runMacro(std::vector<nlohmann::json>, QStringList);

protected slots:
	void actionNeeded();
	void actionNeeded( int );
	void actionNeeded( bool );

protected:

protected:
	QTextEdit* m_recordEdit, * m_macroEdit, * m_filesEdit;
	QPushButton* m_runMacroButton, * m_loadMacroButton, * m_saveMacroButton, * m_transferToRunnerButton, * m_transferToClipboardButton, * m_saveRecorderButton, * m_openFileButton, * m_openDirButton;

	QString m_pathForOpening;

	poca::core::MyObjectInterface* m_object;
	poca::core::MediatorWObjectFWidget * m_mediator;

	std::vector <nlohmann::json> m_jsonRecord, m_jsonRun;
};
#endif // PythonWidget_h__

