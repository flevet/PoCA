/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MacroWidget.cpp
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

#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QColorDialog>
#include <QtGui/QRegExpValidator>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QOpenGLWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QApplication>
#include <QtCore/QDir>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QSplitter>
#include <QtGui/QClipboard>
#include <QtCore/QMimeData>
#include <fstream>
#include <iomanip>

#include <General/Misc.h>
#include <Plot/Icons.hpp>
#include <Geometry/DetectionSet.hpp>
#include <General/PythonInterpreter.hpp>
#include <Objects/MyObject.hpp>
#include <General/Engine.hpp>
#include <General/JsonCommandContext.hpp>

#include "../Widgets/MacroWidget.hpp"

std::string slurp(std::ifstream& in) {
	std::ostringstream sstr;
	sstr << in.rdbuf();
	return sstr.str();
}

MacroWidget::MacroWidget(poca::core::MediatorWObjectFWidget * _mediator, QWidget* _parent/*= 0*/):m_pathForOpening(QDir::currentPath())
{
	m_mediator = _mediator;
	m_object = NULL;

	this->setObjectName("MacroWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");

	m_recordEdit = new QTextEdit;
	m_recordEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_transferToRunnerButton = new QPushButton("Transfer to runner");
	m_transferToRunnerButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_transferToClipboardButton = new QPushButton("Transfer to clipboard");
	m_transferToClipboardButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_saveRecorderButton = new QPushButton("Save recorder");
	m_saveRecorderButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	QWidget* emptyRecorderW = new QWidget;
	emptyRecorderW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QHBoxLayout* layoutLineRecorder = new QHBoxLayout;
	layoutLineRecorder->addWidget(m_transferToRunnerButton);
	layoutLineRecorder->addWidget(m_transferToClipboardButton);
	layoutLineRecorder->addWidget(emptyRecorderW);
	layoutLineRecorder->addWidget(m_saveRecorderButton);
	QVBoxLayout* layout = new QVBoxLayout;
	layout->addWidget(m_recordEdit);
	layout->addLayout(layoutLineRecorder);
	QWidget* recordWidget = new QWidget;
	recordWidget->setLayout(layout);

	int maxSize = 20;
	QSplitter* splitter = new QSplitter(Qt::Vertical, _parent);
	splitter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_macroEdit = new QTextEdit;
	m_macroEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_filesEdit = new QTextEdit;
	m_filesEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_filesEdit->setAcceptDrops(true);
	//m_filesEdit->installEventFilter(this);
	m_filesEdit->viewport()->setAcceptDrops(true);
	m_filesEdit->viewport()->installEventFilter(this);
	splitter->addWidget(m_macroEdit);
	splitter->addWidget(m_filesEdit);
	splitter->setStretchFactor(0, 3);
	splitter->setStretchFactor(1, 1);
	m_loadMacroButton = new QPushButton("Load macro");
	m_loadMacroButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_saveMacroButton = new QPushButton("Save macro");
	m_saveMacroButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_openFileButton = new QPushButton();
	m_openFileButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_openFileButton->setMaximumSize(QSize(maxSize, maxSize));
	m_openFileButton->setIcon(QIcon(QPixmap(poca::plot::openFileIcon)));
	m_openFileButton->setToolTip("Open file");
	QObject::connect(m_openFileButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_openDirButton = new QPushButton();
	m_openDirButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_openDirButton->setMaximumSize(QSize(maxSize, maxSize));
	m_openDirButton->setIcon(QIcon(QPixmap(poca::plot::openDirIcon)));
	m_openDirButton->setToolTip("Open dir");
	QObject::connect(m_openDirButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_addDirButton = new QPushButton();
	m_addDirButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_addDirButton->setMaximumSize(QSize(maxSize, maxSize));
	m_addDirButton->setIcon(QIcon(QPixmap(poca::plot::plusIcon)));
	m_addDirButton->setToolTip("Add dir");
	QObject::connect(m_addDirButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_runMacroButton = new QPushButton("Run");
	m_runMacroButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_runMacroOnFilesButton = new QPushButton("Run on files");
	m_runMacroOnFilesButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_runMacroOnOpenedDataButton = new QPushButton("Run on opened data");
	m_runMacroOnOpenedDataButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	QWidget* emptyRunnerW = new QWidget;
	emptyRunnerW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QHBoxLayout* layoutLineRunner = new QHBoxLayout;
	layoutLineRunner->addWidget(m_loadMacroButton);
	layoutLineRunner->addWidget(m_saveMacroButton);
	layoutLineRunner->addWidget(m_openFileButton);
	layoutLineRunner->addWidget(m_openDirButton);
	layoutLineRunner->addWidget(m_addDirButton);
	layoutLineRunner->addWidget(emptyRunnerW);
	layoutLineRunner->addWidget(m_runMacroButton);
	layoutLineRunner->addWidget(m_runMacroOnFilesButton);
	layoutLineRunner->addWidget(m_runMacroOnOpenedDataButton);
	QVBoxLayout* layoutMacro = new QVBoxLayout;
	layoutMacro->addWidget(splitter);
	layoutMacro->addLayout(layoutLineRunner);
	QWidget* macroWidget = new QWidget;
	macroWidget->setLayout(layoutMacro);

	int index = this->addTab(macroWidget, QObject::tr("Runner"));
	index = this->addTab(recordWidget, QObject::tr("Recorder"));

	QObject::connect(m_transferToRunnerButton, SIGNAL(released()), this, SLOT(actionNeeded()));
	QObject::connect(m_transferToClipboardButton, SIGNAL(released()), this, SLOT(actionNeeded()));
	QObject::connect(m_saveRecorderButton, SIGNAL(released()), this, SLOT(actionNeeded()));

	QObject::connect(m_loadMacroButton, SIGNAL(released()), this, SLOT(actionNeeded()));
	QObject::connect(m_saveMacroButton, SIGNAL(released()), this, SLOT(actionNeeded()));
	QObject::connect(m_runMacroButton, SIGNAL(released()), this, SLOT(actionNeeded()));
	QObject::connect(m_runMacroOnFilesButton, SIGNAL(released()), this, SLOT(actionNeeded()));
	QObject::connect(m_runMacroOnOpenedDataButton, SIGNAL(released()), this, SLOT(actionNeeded()));
}

MacroWidget::~MacroWidget()
{
}

void MacroWidget::actionNeeded()
{
	QObject* sender = QObject::sender();
	if (sender == m_runMacroButton) {
		getJsonsFromTextEdit(m_macroEdit, m_jsonRun);
		emit(runMacro(m_jsonRun, false));
	}
	else if (sender == m_runMacroOnOpenedDataButton) {
		getJsonsFromTextEdit(m_macroEdit, m_jsonRun);
		emit(runMacro(m_jsonRun, true));
	}
	else if (sender == m_runMacroOnFilesButton) {
		if (m_filesEdit->toPlainText().isEmpty()) {
			return;
		}

		getJsonsFromTextEdit(m_macroEdit, m_jsonRun);

		QString files = m_filesEdit->toPlainText();
		if (files.endsWith("\n"))
			files.chop(1);
		QStringList listFiles = files.split("\n");
		emit(runMacro(m_jsonRun, listFiles));
	}
	else if (sender == m_loadMacroButton) {
		QString path = QDir::currentPath(), tmp = path;
		if (!tmp.endsWith("/")) tmp.append("/");
		tmp.append("macros");
		QDir test(tmp);
		if (test.exists())
			path = tmp;
		QString filename = QFileDialog::getOpenFileName(0,
			QObject::tr("Select one file to open"),
			path,
			QObject::tr("Macro file (*.txt)"), 0, QFileDialog::DontUseNativeDialog);
		
		if (filename.isEmpty())
			return;

		m_macroEdit->clear();
		m_jsonRun.clear();
		std::ifstream fs(filename.toStdString());
		QString text = slurp(fs).c_str();
		getJsonsFromQString(text, m_jsonRun);
		for(const auto& json : m_jsonRun)
			m_macroEdit->append(json.dump(4).c_str());
		fs.close();
	}
	else if (sender == m_saveMacroButton || sender == m_saveRecorderButton) {
		QString path = QDir::currentPath(), tmp = path;
		if (!tmp.endsWith("/")) tmp.append("/");
		tmp.append("macros");
		QDir test(tmp);
		if (test.exists())
			path = tmp;
		QString filename = QFileDialog::getSaveFileName(0,
			QObject::tr("Select one file to save"),
			path,
			QObject::tr("Macro file (*.txt)"), 0, QFileDialog::DontUseNativeDialog);

		if (filename.isEmpty())
			return;
		if (!filename.endsWith(".txt"))
			filename.append(".txt");

		QTextEdit* textEdit = sender == m_saveRecorderButton ? m_recordEdit : m_macroEdit;
		std::vector <nlohmann::json>& jsons = sender == m_saveRecorderButton ? m_jsonRecord : m_jsonRun;

		getJsonsFromTextEdit(textEdit, jsons);

		std::ofstream fs(filename.toStdString());
		for (const auto& json : jsons)
			fs << json.dump() << std::endl;
		fs.close();
	}
	else if (sender == m_transferToRunnerButton) {
		m_macroEdit->clear();
		m_macroEdit->setText(m_recordEdit->toPlainText());
		m_jsonRun = m_jsonRecord;
	}
	else if (sender == m_transferToClipboardButton) {
		QClipboard* clipboard = QApplication::clipboard();
		clipboard->setText(m_recordEdit->toPlainText());
	}
	else if (sender == m_openFileButton) {
		QString path = m_pathForOpening;
		QStringList filenames = QFileDialog::getOpenFileNames(0,
			QObject::tr("Select one or more files to open"),
			path,
			QObject::tr("Localization files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

		if (filenames.isEmpty()) return;
		QFileInfo info(filenames[0]);
		m_pathForOpening = info.dir().absolutePath();
		getJsonsFromTextEdit(m_macroEdit, m_jsonRun);
		emit(runMacro(m_jsonRun, filenames));
	}
	else if (sender == m_openDirButton || sender == m_addDirButton) {
		QString path = m_pathForOpening;
		QString dirName = QFileDialog::getExistingDirectory(0,
			QObject::tr("Select directory"),
			path,
			QFileDialog::DontUseNativeDialog | QFileDialog::DontResolveSymlinks);

		if (dirName.isEmpty()) return;

		m_pathForOpening = dirName;

		QDir dir(dirName);
		dir.setFilter(QDir::Files | QDir::NoSymLinks);

		if (!dirName.endsWith("/"))
			dirName.append("/");

		poca::core::Engine* engine = poca::core::Engine::instance();
		const QStringList& availableExtensions = engine->extensions();


		QFileInfoList list = dir.entryInfoList();
		QStringList filenames;
		for (int i = 0; i < list.size(); ++i) {
			QFileInfo fileInfo = list.at(i);
			QString ext = fileInfo.suffix();  // Get the extension WITHOUT the dot
			if (!availableExtensions.contains(ext, Qt::CaseInsensitive)) continue;
			QString filename = fileInfo.fileName();
			filenames.push_back(dirName + filename);
		}
		QString allFiles = sender == m_openDirButton ? QString() : m_filesEdit->toPlainText() + "\n";
		m_filesEdit->setPlainText(allFiles + filenames.join("\n"));
		/*if (filenames.isEmpty()) return;
		getJsonsFromTextEdit(m_macroEdit, m_jsonRun);
		emit(runMacro(m_jsonRun, filenames));*/
	}
}

void MacroWidget::actionNeeded( int _val )
{
	
}

void MacroWidget::actionNeeded(bool _val)
{
	
}

void MacroWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
}

void MacroWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo & _aspect)
{

}

void MacroWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo * _ci)
{
	this->performAction(_wobj, _ci);
}

void MacroWidget::execute(poca::core::CommandInfo* _com)
{
	poca::core::CommandRuntimeContext context;
	execute(_com, context);
}

void MacroWidget::execute(poca::core::CommandInfo* _com, const poca::core::CommandRuntimeContext& _context)
{
	if (_com->nameCommand == "saveParameters") {
		nlohmann::json* json = nullptr;
	if (_context.has<poca::core::JsonFileContext>())
		json = _context.get<poca::core::JsonFileContext>().file;
		if (json == nullptr) return;

		std::string nameStr = objectName().toStdString();
		QString text;
		m_jsonRun.clear();
		getJsonsFromTextEdit(m_macroEdit, m_jsonRun);
		for (auto json : m_jsonRun)
			text.append(json.dump().c_str()).append("\n");
		if (!(*json)[nameStr].contains("macro"))
			json->erase(nameStr);
		(*json)[nameStr]["macro"] = text.toStdString();
		(*json)[nameStr]["path"] = m_pathForOpening.toStdString();
		(*json)[nameStr]["filepaths"] = m_filesEdit->toPlainText().toStdString();
	}
}

void MacroWidget::loadParameters(const nlohmann::json& _json)
{
	std::string nameStr = objectName().toStdString();
	if (_json.contains(nameStr)) {
		m_macroEdit->clear();
		m_jsonRun.clear();
		if (_json[nameStr].contains("macro")) {
			try {
				QString text = _json[nameStr]["macro"].get<string>().c_str();
				std::cout << text.toStdString() << std::endl;
				getJsonsFromQString(text, m_jsonRun);
				for (const auto& json : m_jsonRun)
					m_macroEdit->append(json.dump(4).c_str());
			}
			catch (nlohmann::json::exception& e) {
				std::cout << e.what() << std::endl;
			}
			m_pathForOpening = _json[nameStr]["path"].get<string>().c_str();
		}
		else {
			try {
				QString text = _json[nameStr].get<string>().c_str();
				getJsonsFromQString(text, m_jsonRun);
				for (const auto& json : m_jsonRun)
					m_macroEdit->append(json.dump(4).c_str());
			}
			catch (nlohmann::json::exception& e) {
				std::cout << e.what() << std::endl;
			}
		}
		if (_json[nameStr].contains("filepaths")) {
			QString text = _json[nameStr]["filepaths"].get<string>().c_str();
			std::cout << text.toStdString() << std::endl;
			m_filesEdit->setText(text);
		}
	}
}

void MacroWidget::getJsonsFromQString(const QString& _text, std::vector <nlohmann::json>& _jsons)
{
	_jsons.clear();
	for (QString commandTxt : _text.split("\n")) {
		if (commandTxt.isEmpty()) continue;
		std::stringstream ss;
		ss.str(commandTxt.toStdString());
		nlohmann::json json;
		ss >> json;
		if (json.empty()) continue;
		_jsons.push_back(json);
	}
}

void MacroWidget::getJsonsFromTextEdit(QTextEdit* _textEdit, std::vector <nlohmann::json>& _jsons)
{
	_jsons.clear();
	try {
		QString text = _textEdit->toPlainText();
		text.replace("\n\n", "\n");
		std::stringstream ss;
		ss.str(text.toStdString());
		nlohmann::json js;
		try
		{
			size_t i = 0;
			while (ss.peek() != EOF)
			{
				ss >> js;
				_jsons.push_back(js);
			}
		}
		catch (std::exception& e)
		{
			std::cerr << "    std::exception:" << e.what() << std::endl;
		}
	}
	catch (nlohmann::json::exception& e) {
	}
}

bool MacroWidget::eventFilter(QObject* watched, QEvent* event)
{
	if (watched == m_filesEdit->viewport())
	{
		if (event->type() == QEvent::DragEnter)
		{
			QDragEnterEvent* dragEnterEvent = static_cast<QDragEnterEvent*>(event);
			if (dragEnterEvent->mimeData()->hasUrls())
			{
				dragEnterEvent->acceptProposedAction();
				return true;
			}
		}
		else if (event->type() == QEvent::Drop)
		{
			QDropEvent* dropEvent = static_cast<QDropEvent*>(event);
			QList<QUrl> urlList = dropEvent->mimeData()->urls();

			QStringList filePaths;
			for (const QUrl& url : urlList)
			{
				QString localPath = url.toLocalFile();
				if (!localPath.isEmpty())
				{
					filePaths << localPath;
				}
			}

			m_filesEdit->insertPlainText(filePaths.join("\n"));

			dropEvent->acceptProposedAction();
			return true;
		}
	}

	return QTabWidget::eventFilter(watched, event);
}
