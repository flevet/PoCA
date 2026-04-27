/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonWidget.cpp
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

#ifndef NO_PYTHON

#include <QtWidgets/QLabel>
#include <QtWidgets/QColorDialog>
#include <QtGui/QRegExpValidator>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QOpenGLWidget>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>
#include <QtCore/QVector>
#include <fstream>

#include <General/Misc.h>
#include <Plot/Icons.hpp>
#include <Geometry/DetectionSet.hpp>
#include <General/PythonInterpreter.hpp>
#include <Objects/MyObject.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/ObjectLists.hpp>
#include <General/MyData.hpp>
#include <DesignPatterns/MacroRecorderSingleton.hpp>
#include <General/Engine.hpp>
#include <General/JsonCommandContext.hpp>

#include "../Widgets/PythonWidget.hpp"

PythonWidget::PythonWidget(poca::core::MediatorWObjectFWidget* _mediator, QWidget* _parent/*= 0*/) :QTabWidget(_parent)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;
	m_object = NULL;

	this->setObjectName("PythonWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");

	QGroupBox * groupFileFunction = new QGroupBox(tr("Python execution"));
	groupFileFunction->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	int maxSize = 30;
	m_buttonOpenPython = new QPushButton();
	m_buttonOpenPython->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_buttonOpenPython->setMaximumSize(QSize(maxSize, maxSize));
	m_buttonOpenPython->setIcon(QIcon(QPixmap(poca::plot::openFileIcon)));
	m_buttonOpenPython->setToolTip("Select Python executable");
	QObject::connect(m_buttonOpenPython, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_labelPythonExecutable = new QLabel;
	m_labelPythonExecutable->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QHBoxLayout* layoutPython = new QHBoxLayout;
	layoutPython->addWidget(new QLabel("Python executable:"));
	layoutPython->addWidget(m_buttonOpenPython);
	layoutPython->addWidget(m_labelPythonExecutable);
	m_buttonOpenFile = new QPushButton();
	m_buttonOpenFile->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_buttonOpenFile->setMaximumSize(QSize(maxSize, maxSize));
	m_buttonOpenFile->setIcon(QIcon(QPixmap(poca::plot::openFileIcon)));
	m_buttonOpenFile->setToolTip("Select Python script");
	QObject::connect(m_buttonOpenFile, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_labelPythonFile = new QLabel;
	m_labelPythonFile->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QHBoxLayout* layout1 = new QHBoxLayout;
	layout1->addWidget(new QLabel("Python script:"));
	layout1->addWidget(m_buttonOpenFile);
	layout1->addWidget(m_labelPythonFile);
	QVBoxLayout* layout3 = new QVBoxLayout;
	layout3->addLayout(layoutPython);
	layout3->addLayout(layout1);
	groupFileFunction->setLayout(layout3);


	QGroupBox* groupListFeatures = new QGroupBox(tr("Features to send to Python"));
	groupListFeatures->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	for (auto n = 0; n < 2; n++) {
		m_lists[n] = new QListWidget;
		m_lists[n]->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
		m_lists[n]->setDragDropMode(QAbstractItemView::DragDrop);
		m_lists[n]->setDefaultDropAction(Qt::MoveAction);
		m_lists[n]->setSelectionMode(QAbstractItemView::ExtendedSelection);
	}
	QHBoxLayout* layoutList = new QHBoxLayout;
	layoutList->addWidget(m_lists[0]);
	layoutList->addWidget(m_lists[1]);
	groupListFeatures->setLayout(layoutList);

	m_BCCombo = new QComboBox;
	m_BCCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

	QGroupBox* groupPredefined = new QGroupBox(tr("Add to predefined modules"));
	groupPredefined->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_addToPredefinedModules = new QCheckBox("Yes");
	m_addToPredefinedModules->setChecked(false);
	m_addToPredefinedModules->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* lblAddPredefinedModules = new QLabel("Command name:");
	lblAddPredefinedModules->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_namePredefinedCommand = new QLineEdit;
	m_namePredefinedCommand->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QGridLayout* layoutPredefined = new QGridLayout;
	layoutPredefined->addWidget(m_addToPredefinedModules, 0, 0, 1, 1);
	layoutPredefined->addWidget(lblAddPredefinedModules, 1, 0, 1, 1);
	layoutPredefined->addWidget(m_namePredefinedCommand, 1, 1, 1, 1);
	groupPredefined->setLayout(layoutPredefined);

	m_buttonExecuteScript = new QPushButton("Execute script");
	m_buttonExecuteScript->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QObject::connect(m_buttonExecuteScript, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	QWidget* emptyW = new QWidget;
	emptyW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	QVBoxLayout* layout = new QVBoxLayout;
	//layout->addWidget(m_groupPreloadedPythonFiles);
	layout->addWidget(groupFileFunction);
	groupListFeatures->setVisible(false);
	layout->addWidget(groupListFeatures);
	layout->addWidget(groupPredefined);
	layout->addWidget(m_buttonExecuteScript, Qt::AlignRight);
	layout->addWidget(emptyW);
	//this->setLayout(layout);
	//this->setMinimumHeight(150);
	//this->setMaximumHeight(800);

	/*QVBoxLayout* layout = new QVBoxLayout;
	layout->addWidget(m_groupPreloadedPythonFiles);
	layout->addWidget(emptyW);*/

	//this->setLayout(layout);

	QWidget* loadPythonFileWidget = new QWidget;
	loadPythonFileWidget->setLayout(layout);
	//loadPythonFileWidget->setMinimumHeight(150);
	//loadPythonFileWidget->setMaximumHeight(800);
	int index = this->addTab(loadPythonFileWidget, QObject::tr("Run Python file"));


	QObject::connect(m_BCCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(actionNeeded(int)));
}

PythonWidget::~PythonWidget()
{
}

void PythonWidget::populatePredefinedButtons()
{
	int maxSize = 100;
	std::vector <std::vector<poca::core::CommandInfo>::iterator> toErase;
	for (std::vector<poca::core::CommandInfo>::iterator com = m_pythonCommands.begin(); com != m_pythonCommands.end(); com++) {
		const std::string& filename = com->getParameter<std::string>("filename");

		if (!poca::core::file_exists(filename)) {
			//if the file does not exist, we remove the command from the vector
			toErase.insert(toErase.begin(), com);
			continue;
		}
	}
	for (auto it : toErase)
		m_pythonCommands.erase(it);

	m_layoutPredefined = new QGridLayout;

	for(auto n = 0; n < m_pythonCommands.size(); n++)
		addPredefinedButton(n);

	QWidget* emptyW = new QWidget;
	emptyW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	QVBoxLayout* layout = new QVBoxLayout;
	layout->addLayout(m_layoutPredefined);
	layout->addWidget(emptyW);
	
	QWidget* preDefinedWidget = new QWidget;
	preDefinedWidget->setLayout(layout);
	int index = this->addTab(preDefinedWidget, QObject::tr("Predefined modules"));
}

void PythonWidget::addPredefinedButton(uint32_t _indexCommand)
{
	int maxSize = 100, maxSize2 = 20;
	QPushButton* button = new QPushButton;
	button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	button->setMaximumSize(QSize(maxSize, maxSize));
	button->setIconSize(QSize(maxSize, maxSize));;
	button->setIcon(QIcon(QPixmap(poca::plot::filePythonIcon)));
	QObject::connect(button, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_buttonsPreloaded.push_back(button);

	QPushButton* buttonRemove = new QPushButton;
	buttonRemove->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	buttonRemove->setMaximumSize(QSize(maxSize2, maxSize2));
	buttonRemove->setIconSize(QSize(maxSize2, maxSize2));;
	buttonRemove->setIcon(QIcon(QPixmap(poca::plot::bin2Icon)));
	buttonRemove->setCheckable(true);
	m_buttonsRemovePreloaded.push_back(buttonRemove);

	QGridLayout* layout = new QGridLayout;
	layout->addWidget(m_buttonsPreloaded[_indexCommand], 0, 0, 1, 2);
	layout->setAlignment(m_buttonsPreloaded[_indexCommand], Qt::AlignHCenter);
	std::string label = m_pythonCommands[_indexCommand].hasParameter("buttonLabel") ? m_pythonCommands[_indexCommand].getParameter<std::string>("buttonLabel") : m_pythonCommands[_indexCommand].getNameCommand();
	QLabel* lbl = new QLabel(label.c_str());
	lbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	layout->addWidget(lbl, 1, 0, 1, 1);
	layout->setAlignment(lbl, Qt::AlignLeft);
	layout->addWidget(m_buttonsRemovePreloaded[_indexCommand], 1, 1, 1, 1);
	layout->setAlignment(m_buttonsRemovePreloaded[_indexCommand], Qt::AlignRight);
	QWidget* w = new QWidget;
	w->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	w->setMaximumWidth(100);
	w->setLayout(layout);

	if (m_curColumn == 3) {
		m_curRow++;
		m_curColumn = 0;
	}
	m_layoutPredefined->addWidget(w, m_curRow, m_curColumn++);
}

void PythonWidget::actionNeeded()
{
	QObject* sender = QObject::sender();
	bool found = false;
	for (size_t n = 0; n < m_buttonsPreloaded.size() && !found; n++) {
		found = (m_buttonsPreloaded[n] == sender);
		if (found) {
			poca::core::CommandInfo command(m_pythonCommands[n]);
			command.recordable = true;
			execute(&command);
		}
	}
	if (sender == m_buttonOpenPython) {
		QString filename = QFileDialog::getOpenFileName(0,
			QObject::tr("Select Python executable"),
			QDir::currentPath(),
			QObject::tr("Python executable (python.exe);;Executable (*.exe)"), 0, QFileDialog::DontUseNativeDialog);
		if (filename.isEmpty()) return;
		m_labelPythonExecutable->setText(filename);
	}
	else if (sender == m_buttonOpenFile) {
		QString filename = QFileDialog::getOpenFileName(0,
			QObject::tr("Select one Python file"),
			QDir::currentPath(),
			QObject::tr("Python file (*.py)"), 0, QFileDialog::DontUseNativeDialog);

		if (filename.isEmpty()) return;
		m_labelPythonFile->setText(filename);
	}
	else if (sender == m_buttonExecuteScript) {
		if (m_labelPythonFile->text().isEmpty()) {
			QMessageBox msgBox;
			msgBox.setText("Please choose a Python script before execution.");
			msgBox.exec();
			return;
		}
		QString name = m_labelPythonFile->text();
		name = name.right(name.size() - name.lastIndexOf("/") - 1);
		std::string nameModule = name.left(name.lastIndexOf(".")).toStdString(), filename = name.toStdString(), commandName = nameModule;
		if (m_addToPredefinedModules->isChecked()) {
			commandName = m_namePredefinedCommand->text().toStdString();
			if (commandName.empty()) {
				QMessageBox msgBox;
				msgBox.setText("Error: when adding a Python command to predefined modules, it is mandatory to set the command name.");
				msgBox.exec();
				return;
			}
		}
		poca::core::CommandInfo com(true, commandName,
			"pythonExecutable", m_labelPythonExecutable->text().toStdString(),
			"filename", m_labelPythonFile->text().toStdString());

		execute(&com);
		if (m_addToPredefinedModules->isChecked()) {
			for (auto n = 0; n < m_pythonCommands.size(); n++) {
				if (m_pythonCommands[n].getNameCommand() == com.getNameCommand()) {
					m_pythonCommands[n] = com;
					QMessageBox msgBox;
					msgBox.setText("The command " + QString(com.getNameCommand().c_str()) + " was already defined in the predefined modules and was updated.");
					msgBox.exec();
					return;
				}
			}
			m_pythonCommands.push_back(com);
			addPredefinedButton(m_pythonCommands.size() - 1);
		}
	}
}

void PythonWidget::actionNeeded(int _idx)
{
	if (_idx == -1) return;
	/*QObject* sender = QObject::sender();
	bool found = false;
	if (sender == m_BCCombo) {
		poca::core::BasicComponentInterface* bc = m_object->getBasicComponent(_idx);
		m_lists[0]->clear();
		populateListWidget(bc, m_lists[0]);
	}*/
}

void PythonWidget::actionNeeded(bool _val)
{

}

void PythonWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
}

void PythonWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	poca::core::MyObjectInterface* objOneColor = obj->currentObject();
	if (objOneColor == NULL) {
		//m_groupPreloadedPythonFiles->setVisible(false);
		return;
	}
	//m_groupPreloadedPythonFiles->setVisible(true);

	m_object = obj;

	bool visible = true;// (objOneColor != NULL && objOneColor->hasBasicComponent("DetectionSet"));
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
#endif

	m_lists[0]->clear();

	m_object = obj;

	if (m_object->nbBasicComponents() < 1) return;

	if (_aspect == "LoadObjCharacteristicsAllWidgets") {

		//poca::core::BasicComponentInterface* bci = obj->getBasicComponent("DetectionSet");
		//if (!bci) return;
		m_BCCombo->clear();
		for (size_t n = 0; n < m_object->nbBasicComponents(); n++) {
			m_BCCombo->insertItem(n, m_object->getBasicComponent(n)->getName().c_str());
			poca::core::BasicComponentInterface* bc = m_object->getBasicComponent(n);
			populateListWidget(bc, m_lists[0]);
		}
	}
}

void PythonWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

void PythonWidget::populateListWidget(poca::core::BasicComponentInterface* _bc, QListWidget* _listW)
{
	if (_bc->nbComponents() == 0) return;
	QString nameBC = _bc->getName().c_str();
	for (std::string component : _bc->getNameData())
		_listW->addItem(nameBC + " -> " + component.c_str());
}

void PythonWidget::execute(poca::core::CommandInfo* _com)
{
	poca::core::CommandExecutionContext context;
	execute(_com, context);
}

void PythonWidget::execute(poca::core::CommandInfo* _com, const poca::core::CommandExecutionContext& _context)
{
	if (_com->nameCommand == "saveParameters") {
		nlohmann::json* json = nullptr;
	if (_context.has<poca::core::JsonFileContext>())
		json = _context.get<poca::core::JsonFileContext>().file;
		if (json == nullptr) return;

		std::vector <nlohmann::json> commands;
		for(auto n = 0; n < m_pythonCommands.size(); n++)
			if(!m_buttonsRemovePreloaded[n]->isChecked())
				commands.push_back(m_pythonCommands[n].json);

		std::string nameStr = objectName().toStdString();
		(*json)[nameStr] = commands;
			if (!m_labelPythonExecutable->text().isEmpty())
				(*json)["PythonParameters"]["python_executable_path"] = m_labelPythonExecutable->text().toStdString();
	}
	if(_com->hasParameter("filename"))
		executePythonScript(*_com);

	if (_com->isRecordable())
		poca::core::MacroRecorderSingleton::instance()->addCommand("PythonWidget", _com);
}

void PythonWidget::loadParameters(const nlohmann::json& _json)
{
	if (_json.contains("PythonParameters") && _json["PythonParameters"].contains("python_executable_path"))
		m_labelPythonExecutable->setText(_json["PythonParameters"]["python_executable_path"].get<std::string>().c_str());

	std::string nameStr = objectName().toStdString();
	if (_json.contains(nameStr)) {
		try {
			std::vector <nlohmann::json> commands = _json[nameStr].get<std::vector <nlohmann::json>>();
			for (const auto& json : commands) {
				for (auto& [nameCommand, value] : json.items()) {
					poca::core::CommandInfo command = createCommand(nameCommand, json[nameCommand]);
					if (!command.empty()) {
						m_pythonCommands.push_back(command);
					}
				}
			}
		}
		catch (nlohmann::json::exception& e) {
			std::cout << e.what() << std::endl;
		}
	}

	populatePredefinedButtons();
}

poca::core::CommandInfo PythonWidget::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	const poca::core::CommandSpec spec(_nameCommand, {
		{ "pythonExecutable", poca::core::CommandParameterType::String },
		{ "filename", poca::core::CommandParameterType::String, true },
		{ "features", poca::core::CommandParameterType::Array },
		{ "buttonLabel", poca::core::CommandParameterType::String }
	});
	return spec.create(false, _parameters);
}

namespace {
	bool collectPocaPythonInputsFromFeatures(poca::core::MyObjectInterface* _obj, const std::vector<std::string>& _features, const std::string& _filename, const std::string& _nameFunction, std::vector<poca::core::PythonInterpreter::PythonFeatureInput>& _inputs)
	{
		if (_obj == nullptr) return false;
		for (const auto& feature : _features) {
			auto pos = feature.find(" -> ");
			if (pos == std::string::npos) {
				std::cout << "Error: bad Python feature specification '" << feature << "'. Expected 'Component -> feature'." << std::endl;
				return false;
			}
			std::string comp = feature.substr(0, pos), feat = feature.substr(pos + 4);
			poca::core::BasicComponent* bc = static_cast<poca::core::BasicComponent*>(_obj->getBasicComponent(comp));
			if (!bc) {
				std::cout << "Error: execution of the Python function " << _nameFunction << " from the file " << _filename << " failed. Component " << comp << " does not exist." << std::endl;
				return false;
			}
			if (!bc->hasData(feat)) {
				std::cout << "Error: execution of the Python function " << _nameFunction << " from the file " << _filename << " failed. Feature " << feat << " from component " << comp << " does not exist." << std::endl;
				return false;
			}
			poca::core::PythonInterpreter::PythonFeatureInput input;
			input.component = comp;
			input.feature = feat;
			input.values = &bc->getData<float>(feat);
			_inputs.push_back(input);
		}
		return true;
	}

	bool collectPocaPythonInputsFromRequirements(poca::core::MyObjectInterface* _obj, const nlohmann::json& _description, const std::string& _filename, std::vector<poca::core::PythonInterpreter::PythonFeatureInput>& _inputs)
	{
		if (_obj == nullptr) return false;
		if (!_description.value("ok", false)) {
			std::cout << "Error: could not read PoCA Python script requirements for " << _filename << "." << std::endl;
			if (_description.contains("error")) std::cout << _description["error"].get<std::string>() << std::endl;
			if (_description.contains("traceback")) std::cout << _description["traceback"].get<std::string>() << std::endl;
			return false;
		}
		if (!_description.contains("requirements") || !_description["requirements"].is_array()) {
			std::cout << "Error: Python script " << _filename << " did not return a valid requirements array." << std::endl;
			return false;
		}
		for (const auto& requirement : _description["requirements"]) {
			const std::string comp = requirement.value("component", std::string("DetectionSet"));
			poca::core::BasicComponent* bc = static_cast<poca::core::BasicComponent*>(_obj->getBasicComponent(comp));
			if (!bc) {
				std::cout << "Error: Python script " << _filename << " requires component '" << comp << "', but this component does not exist." << std::endl;
				return false;
			}
			if (!requirement.contains("features") || !requirement["features"].is_array()) {
				std::cout << "Error: Python script " << _filename << " has an invalid requirement for component '" << comp << "'." << std::endl;
				return false;
			}
			for (const auto& featureRequirement : requirement["features"]) {
				const std::string feat = featureRequirement.value("name", std::string());
				const bool optional = featureRequirement.value("optional", false);
				if (feat.empty()) {
					std::cout << "Error: Python script " << _filename << " has a feature requirement without a name." << std::endl;
					return false;
				}
				if (!bc->hasData(feat)) {
					if (optional) continue;
					std::cout << "Error: Python script " << _filename << " requires feature '" << feat << "' from component '" << comp << "', but it does not exist." << std::endl;
					return false;
				}
				poca::core::PythonInterpreter::PythonFeatureInput input;
				input.component = comp;
				input.feature = feat;
				input.values = &bc->getData<float>(feat);
				_inputs.push_back(input);
			}
		}
		if (_inputs.empty())
			std::cout << "Warning: Python script " << _filename << " declared no PoCA input features." << std::endl;
		return true;
	}

	bool applyPocaPythonActions(poca::core::MyObjectInterface* _obj, const nlohmann::json& _response)
	{
		if (!_response.contains("actions")) return true;
		bool changedObject = false;
		for (const auto& action : _response["actions"]) {
			const std::string type = action.value("type", std::string());
			if (type == "display") {
				std::cout << action.value("text", std::string()) << std::endl;
			}
			else if (type == "add_feature") {
				const std::string component = action.value("component", std::string());
				const std::string feature = action.value("feature", std::string());
				if (component.empty() || feature.empty() || !action.contains("values")) {
					std::cout << "Error: Python add_feature action requires component, feature and values." << std::endl;
					return false;
				}
				poca::core::BasicComponentInterface* bc = _obj->getBasicComponent(component);
				if (!bc) {
					std::cout << "Error: Python requested adding feature " << feature << " to unknown component " << component << "." << std::endl;
					return false;
				}
				std::vector<float> newFeature;
				newFeature.reserve(action["values"].size());
				for (const auto& v : action["values"])
					newFeature.push_back(static_cast<float>(v.get<double>()));
				if (newFeature.size() != bc->nbElements()) {
					std::cout << "Error: Python feature \"" << feature << "\" has " << newFeature.size() << " values, but component \"" << component << "\" has " << bc->nbElements() << " elements." << std::endl;
					return false;
				}
				bc->addFeature(feature, poca::core::generateDataWithLog(newFeature));
				std::cout << "Python added feature '" << feature << "' to component '" << component << "' (" << newFeature.size() << " values)." << std::endl;
				changedObject = true;
			}
			else if (type == "create_dataset") {
				std::cout << "Python requested creation of dataset '" << action.value("name", std::string("unnamed")) << "'. This action is described in JSON but is not connected to PoCA object creation yet." << std::endl;
			}
			else {
				std::cout << "Warning: unknown Python action type '" << type << "'." << std::endl;
			}
		}
		if (changedObject)
			_obj->notify("LoadObjCharacteristicsAllWidgets");
		return true;
	}
}

void PythonWidget::executePythonScript(const poca::core::CommandInfo& _command)
{
	std::vector <std::string> features;
	if (_command.hasParameter("features"))
		features = _command.getParameter< std::vector <std::string>>("features");
	std::string filename = _command.getParameter<std::string>("filename");
	std::string pythonExecutable;
	if (_command.hasParameter("pythonExecutable"))
		pythonExecutable = _command.getParameter<std::string>("pythonExecutable");
	if (pythonExecutable.empty()) {
		nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
		if (parameters.contains("PythonParameters") && parameters["PythonParameters"].contains("python_executable_path"))
			pythonExecutable = parameters["PythonParameters"]["python_executable_path"].get<std::string>();
	}
	if (pythonExecutable.empty()) {
		std::cout << "Error: no Python executable is configured. Set PythonParameters/python_executable_path in poca.ini or select python.exe in the Python widget." << std::endl;
		return;
	}
	const std::string nameFunction = "run";

	poca::core::PythonInterpreter* py = poca::core::PythonInterpreter::instance();
	std::vector<poca::core::PythonInterpreter::PythonFeatureInput> inputs;
	nlohmann::json description;
	if (py->describePocaScript(description, pythonExecutable.c_str(), filename.c_str()) == EXIT_SUCCESS && description.contains("requirements") && !description["requirements"].empty()) {
		if (!collectPocaPythonInputsFromRequirements(m_object->currentObject(), description, filename, inputs))
			return;
	}
	else {
		if (!features.empty()) {
			if (!collectPocaPythonInputsFromFeatures(m_object->currentObject(), features, filename, nameFunction, inputs))
				return;
		}
		else {
			std::cout << "Error: Python script " << filename << " does not declare POCA_INPUTS/poca_inputs(), and no legacy GUI feature list is available." << std::endl;
			return;
		}
	}

	nlohmann::json response;
	bool res = py->executePocaScript(response, inputs, pythonExecutable.c_str(), filename.c_str());
	if (res == EXIT_FAILURE) {
		std::cout << "ERROR! Function run(poca) from Python file " << filename << " was not run." << std::endl;
		return;
	}

	applyPocaPythonActions(m_object->currentObject(), response);
}

#endif
