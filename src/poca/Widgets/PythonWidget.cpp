/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonWidget.cpp
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

#ifndef NO_PYTHON

#include <QtWidgets/QLabel>
#include <QtWidgets/QColorDialog>
#include <QtGui/QRegExpValidator>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QOpenGLWidget>
#include <QtWidgets/QButtonGroup>
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
#include <Geometry/ObjectList.hpp>
#include <General/MyData.hpp>
#include <DesignPatterns/MacroRecorderSingleton.hpp>
#include <DesignPatterns/StateSoftwareSingleton.hpp>

#include "../Widgets/PythonWidget.hpp"

PythonWidget::PythonWidget(poca::core::MediatorWObjectFWidget* _mediator, QWidget* _parent/*= 0*/) :QTabWidget(_parent)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;
	m_object = NULL;

	/*std::vector <std::string> features = {"DetectionSet -> x", "DetectionSet -> y"};
	m_pythonCommands.push_back(poca::core::CommandInfo(false, "nena", "filename", std::string("nena.py"), "nameFunction", std::string("nena"), "features", features, "resultType", std::string("singleValue")));
	m_pythonCommands.push_back(poca::core::CommandInfo(false, "CAML", "buttonLabel", std::string("CAML2D"), "filename", std::string("CAML.py"), "nameFunction", std::string("testCAML2D"), "features", features, "resultType", std::string("addFeature"), "addToComponent", std::string("DetectionSet"), "nameNewFeature", std::string("camID")));
	features.push_back("DetectionSet -> z");
	m_pythonCommands.push_back(poca::core::CommandInfo(false, "CAML", "buttonLabel", std::string("CAML3D"), "filename", std::string("CAML.py"), "nameFunction", std::string("testCAML3D"), "features", features, "resultType", std::string("addFeature"), "addToComponent", std::string("DetectionSet"), "nameNewFeature", std::string("camID")));
	m_pythonCommands.push_back(poca::core::CommandInfo(false, "ellipsoidFit", "buttonLabel", std::string("Locs: fit ellipsoid"), "filename", std::string("ellipsoidFit.py"), "nameFunction", std::string("ls_ellipsoid"), "features", features, "resultType", std::string("singleValue")));
	*/
	this->setObjectName("PythonWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");

	QGroupBox * groupFileFunction = new QGroupBox(tr("Python file & function"));
	groupFileFunction->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	int maxSize = 30;
	m_buttonOpenFile = new QPushButton();
	m_buttonOpenFile->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_buttonOpenFile->setMaximumSize(QSize(maxSize, maxSize));
	m_buttonOpenFile->setIcon(QIcon(QPixmap(poca::plot::openFileIcon)));
	m_buttonOpenFile->setToolTip("Save detections");
	QObject::connect(m_buttonOpenFile, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_labelPythonFile = new QLabel;
	m_labelPythonFile->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QHBoxLayout* layout1 = new QHBoxLayout;
	layout1->addWidget(m_buttonOpenFile);
	layout1->addWidget(m_labelPythonFile);
	QLabel* lblCombo = new QLabel("Identified functions:");
	lblCombo->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_functionNameCombo = new QComboBox;
	m_functionNameCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QObject::connect(m_functionNameCombo, SIGNAL(activated(int)), this, SLOT(updateChosenFunctionName(int)));
	QHBoxLayout* layoutCombo = new QHBoxLayout;
	layoutCombo->addWidget(lblCombo);
	layoutCombo->addWidget(m_functionNameCombo);
	QLabel* lbl1 = new QLabel("Function name (no parameters):");
	lbl1->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_editNameFunction = new QLineEdit;
	m_editNameFunction->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QHBoxLayout* layout2 = new QHBoxLayout;
	layout2->addWidget(lbl1);
	layout2->addWidget(m_editNameFunction);
	QVBoxLayout* layout3 = new QVBoxLayout;
	layout3->addLayout(layout1);
	layout3->addLayout(layoutCombo);
	layout3->addLayout(layout2);
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

	QGroupBox* groupReturnValues = new QGroupBox(tr("Python function return type"));
	groupReturnValues->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_singleValCBox = new QCheckBox("Display results");
	m_singleValCBox->setChecked(true);
	m_singleValCBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_addFeatureCBox = new QCheckBox("Add feature to component"); 
	m_addFeatureCBox->setChecked(false);
	m_addFeatureCBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_createNewDatasetCBox = new QCheckBox("Create new dataset");
	m_createNewDatasetCBox->setChecked(false);
	m_createNewDatasetCBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_createNewDatasetCBox->setEnabled(false);
	m_bgroupGrid = new QButtonGroup;
	m_bgroupGrid->addButton(m_singleValCBox, 0);
	m_bgroupGrid->addButton(m_addFeatureCBox, 1);
	m_bgroupGrid->addButton(m_createNewDatasetCBox, 2);
	m_BCCombo = new QComboBox;
	m_BCCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QLabel* lblNameFeature = new QLabel("Name new feature");
	lblNameFeature->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_nameFeatureEdit = new QLineEdit;
	m_nameFeatureEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QLabel* lblNamDataset = new QLabel("Name new dataset");
	lblNamDataset->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	lblNamDataset->setEnabled(false);
	m_nameNewDatasetEdit = new QLineEdit;
	m_nameNewDatasetEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_nameNewDatasetEdit->setEnabled(false);
	QLabel* lbl = new QLabel("Feature name:");
	lbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QGridLayout* layoutType = new QGridLayout;
	layoutType->addWidget(m_singleValCBox, 0, 0, 1, 1);
	layoutType->addWidget(m_addFeatureCBox, 1, 0, 1, 1);
	layoutType->addWidget(m_BCCombo, 1, 1, 1, 1);
	layoutType->addWidget(lblNameFeature, 1, 2, 1, 1);
	layoutType->addWidget(m_nameFeatureEdit, 1, 3, 1, 1);
	layoutType->addWidget(m_createNewDatasetCBox, 2, 0, 1, 1);
	layoutType->addWidget(lblNamDataset, 2, 2, 1, 1);
	layoutType->addWidget(m_nameNewDatasetEdit, 2, 3, 1, 1);
	groupReturnValues->setLayout(layoutType);

	QGroupBox* groupPredefined = new QGroupBox(tr("Add to predefined modules"));
	groupReturnValues->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
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
	layout->addWidget(groupListFeatures);
	layout->addWidget(groupReturnValues);
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
	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	nlohmann::json& parameters = sss->getParameters();
	std::vector <std::string> names = { "python_path", "python_dll_path", "python_lib_path", "python_packages_path", "python_scripts_path" };
	std::vector <std::string> paths(names.size());
	if (!parameters.contains("PythonParameters")) {
		/*QMessageBox msgBox;
		msgBox.setText("Cannot load predefined modules. Please make sure that the Python paths have been initialized (in Menu >> Plugins >> Python).");
		msgBox.exec();*/
		return;
	}
	if (!parameters["PythonParameters"].contains("python_scripts_path")) {
		/*QMessageBox msgBox;
		msgBox.setText("Cannot load predefined modules. Please make sure that the Python paths to the custom Python scripts has been initialized (in Menu >> Plugins >> Python).");
		msgBox.exec();*/
		return;
	}
	std::string pathToScripts = parameters["PythonParameters"]["python_scripts_path"].get<std::string>();
	if (pathToScripts[pathToScripts.size() - 1] != '/')
		pathToScripts.append("/");

	int maxSize = 100;
	std::vector <std::vector<poca::core::CommandInfo>::iterator> toErase;
	for (std::vector<poca::core::CommandInfo>::iterator com = m_pythonCommands.begin(); com != m_pythonCommands.end(); com++) {
		const std::string& filename = pathToScripts + com->getParameter<std::string>("filename");

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
	if (sender == m_buttonOpenFile) {
		QString path;
		poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
		nlohmann::json& parameters = sss->getParameters();
		if (parameters.contains("PythonParameters") && parameters["PythonParameters"].contains("python_scripts_path")) {
			std::string pathToScripts = parameters["PythonParameters"]["python_scripts_path"].get<std::string>();
			path = pathToScripts.c_str();
		}
		else
			path = QDir::currentPath();

		QString filename = QFileDialog::getOpenFileName(0,
			QObject::tr("Select one Python file"),
			path,
			QObject::tr("Python file (*.py)"), 0, QFileDialog::DontUseNativeDialog);

		if (filename.isEmpty()) return;
		QString pathFile = filename.left(filename.lastIndexOf("/"));
		std::cout << pathFile.toStdString() << std::endl;
		if (pathFile != path) {
			QMessageBox msgBox;
			msgBox.setText("Mandatory: the chosen Python file is required to be inside the Python script folder (" + path + ") defined in PoCA(in Menu >> Plugins >> Python).");
			msgBox.exec();
			return;
		}
		m_labelPythonFile->setText(filename);

		QFile file(filename);
		if (!file.open(QFile::ReadOnly | QFile::Text)) return;
		QStringList functions = identifyPythonFunctionNames(file.readAll());
		m_functionNameCombo->clear();
		m_functionNameCombo->addItems(functions);
	}
	else if (sender == m_buttonExecuteScript) {
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
		std::string nameFunction = m_editNameFunction->text().toStdString();
		std::vector <std::string> features;
		for (auto n = 0; n < m_lists[1]->count(); n++)
			features.push_back(m_lists[1]->item(n)->text().toStdString());
		std::string resultType, addToComponent, nameNewFeature;
		auto idx = m_bgroupGrid->checkedId();
		poca::core::CommandInfo com;
		switch (idx) {
		case 0:
		{
			resultType = "singleValue";
			com = poca::core::CommandInfo(true, commandName, "filename", filename, "nameFunction", nameFunction, "features", features, "resultType", resultType);
			break;
		}
		case 1:
		{
			resultType = "addFeature";
			addToComponent = m_BCCombo->currentText().toStdString();
			nameNewFeature = m_nameFeatureEdit->text().toStdString();
			com = poca::core::CommandInfo(true, commandName, "filename", filename, "nameFunction", nameFunction, "features", features, "resultType", resultType, "addToComponent", addToComponent, "nameNewFeature", nameNewFeature);
			break;
		}
		case 2:
			//resultType = "singleValue";
			break;
		}
		if (resultType.empty())
			return;

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
		poca::core::BasicComponent* bc = m_object->getBasicComponent(_idx);
		m_lists[0]->clear();
		populateListWidget(bc, m_lists[0]);
	}*/
}

void PythonWidget::actionNeeded(bool _val)
{

}

void PythonWidget::executeNena()
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponent* bci = obj->getBasicComponent("DetectionSet");
	poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
	if (dset == NULL) return;

	if (!dset->hasData("frame")) return;

	const std::vector <float>& xs = dset->getData("x");
	const std::vector <float>& ys = dset->getData("y");
	const std::vector <float>& zs = dset->hasData("z") ? dset->getData("z") : std::vector <float>(xs.size(), 0.f);
	const std::vector <float>& times = dset->getData("frame");

	//Prepare data per frame
	std::vector <uint32_t> pointsPerFrame;
	float currentTime = times[0];
	uint32_t n = 0;
	pointsPerFrame.push_back(n++);
	for (; n < dset->nbPoints(); n++) {
		if (times[n] != currentTime) {
			currentTime = times[n];
			pointsPerFrame.push_back(n);
		}
	}
	pointsPerFrame.push_back(n - 1);

	QVector <double> distancesNena;

	double dmin = std::numeric_limits < double >::max(), d = 0.;
	for (size_t currentSlice = 0; currentSlice < pointsPerFrame.size() - 2; currentSlice++) {
		size_t nextSlice = currentSlice + 1;
		for (uint32_t i = pointsPerFrame[currentSlice]; i < pointsPerFrame[nextSlice]; i++) {
			dmin = std::numeric_limits < double >::max();
			bool found = false;
			for (uint32_t j = pointsPerFrame[nextSlice]; j < pointsPerFrame[nextSlice + 1]; j++) {
				d = poca::geometry::distance3DSqr(xs[i], ys[i], zs[i], xs[j], ys[j], zs[j]);
				if (d < dmin) {
					dmin = d;
					found = true;
				}
			}
			if (found) {
				distancesNena.push_back(sqrt(dmin));
			}
		}
	}
	poca::core::PythonInterpreter* py = poca::core::PythonInterpreter::instance();
	QVector <QVector <double>> distances;
	distances.push_back(distancesNena);
	QVector <double> coeffs;
	bool res = py->applyFunctionWithNArraysParameterAnd1ArrayReturned(coeffs, distances, "nena", "nena");
	if (res == EXIT_FAILURE)
		std::cout << "ERROR! NeNA was not run with error message: python script was not run." << std::endl;
	else if(coeffs.empty())
		std::cout << "ERROR! running NeNA failed." << std::endl;
	else
		std::cout << "The average lecalization accuracy by NeNA is at " << coeffs[0] << " nm." << std::endl;
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

	if (m_object->nbBasicComponents() <= 1) return;

	if (_aspect == "LoadObjCharacteristicsAllWidgets") {

		//poca::core::BasicComponent* bci = obj->getBasicComponent("DetectionSet");
		//if (!bci) return;
		m_BCCombo->clear();
		for (size_t n = 0; n < m_object->nbBasicComponents(); n++) {
			m_BCCombo->insertItem(n, m_object->getBasicComponent(n)->getName().c_str());
			poca::core::BasicComponent* bc = m_object->getBasicComponent(n);
			populateListWidget(bc, m_lists[0]);
		}
	}
}

void PythonWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

void PythonWidget::populateListWidget(poca::core::BasicComponent* _bc, QListWidget* _listW)
{
	QString nameBC = _bc->getName().c_str();
	for (std::string component : _bc->getNameData())
		_listW->addItem(nameBC + " -> " + component.c_str());
}

void PythonWidget::execute(poca::core::CommandInfo* _com)
{
	if (_com->nameCommand == "saveParameters") {
		if (!_com->hasParameter("file")) return;
		nlohmann::json* json = _com->getParameterPtr<nlohmann::json>("file");

		std::vector <nlohmann::json> commands;
		for(auto n = 0; n < m_pythonCommands.size(); n++)
			if(!m_buttonsRemovePreloaded[n]->isChecked())
				commands.push_back(m_pythonCommands[n].json);

		std::string nameStr = objectName().toStdString();
		(*json)[nameStr] = commands;
	}
	if(_com->hasParameter("filename") && _com->hasParameter("features") && _com->hasParameter("resultType") && _com->hasParameter("nameFunction") ){
		std::string resultType = _com->getParameter<std::string>("resultType");
		if (resultType == "singleValue")
			executePythonScriptDisplayReturn(*_com);
		else if (resultType == "addFeature")
			executePythonScriptAddFeatureToComponent(*_com);
	}

	if (_com->isRecordable())
		poca::core::MacroRecorderSingleton::instance()->addCommand("PythonWidget", _com);
}

void PythonWidget::loadParameters(const nlohmann::json& _json)
{
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
	poca::core::CommandInfo ci(false, _nameCommand);
	if (!_parameters.contains("filename") || !_parameters.contains("features") || !_parameters.contains("resultType") || !_parameters.contains("nameFunction"))
		return poca::core::CommandInfo();
	ci.addParameter("filename", _parameters["filename"].get<std::string>());
	ci.addParameter("features", _parameters["features"].get<std::vector<std::string>>());
	ci.addParameter("resultType", _parameters["resultType"].get<std::string>());
	if (_parameters.contains("addToComponent"))
		ci.addParameter("addToComponent", _parameters["addToComponent"].get<std::string>());
	if (_parameters.contains("nameNewFeature"))
		ci.addParameter("nameNewFeature", _parameters["nameNewFeature"].get<std::string>());
	if (_parameters.contains("nameFunction"))
		ci.addParameter("nameFunction", _parameters["nameFunction"].get<std::string>());
	if (_parameters.contains("buttonLabel"))
		ci.addParameter("buttonLabel", _parameters["buttonLabel"].get<std::string>());
	return ci;
}

void PythonWidget::executePythonScriptDisplayReturn(const poca::core::CommandInfo& _command)
{
	std::string nameFunction = _command.getParameter<std::string>("nameFunction");

	if (nameFunction == "nena") {
		//This is a hack. This version of nena requires computation of the distances between the locs in the consecutive frames
		//Therefore it needs currently to be done in poca, requiring highjacking the normal run of the python script
		//In a perfect world, this distance computation should be done in the python script
		executeNena();
		return;
	}

	QVector <QVector <double>> dataPython;
	std::vector <std::string> features = _command.getParameter< std::vector <std::string>>("features");
	std::string filename = _command.getParameter<std::string>("filename");

	poca::core::MyObjectInterface* obj = m_object->currentObject();
	for (const auto& feature : features) {
		auto pos = feature.find(" -> ");
		std::string comp = feature.substr(0, pos), feat = feature.substr(pos + 4);
		//std::cout << comp << " ------ " << feat << std::endl;
		poca::core::BasicComponent* bc = obj->getBasicComponent(comp);
		if (!bc) {
			std::cout << "Error: execution of the Python function " << nameFunction << " from the file " << filename << " failed. Component " << comp << " does not exist." << std::endl;
			return;
		}
		if (!bc->hasData(feat)) {
			std::cout << "Error: execution of the Python function " << nameFunction << " from the file " << filename << " failed. Feature " << feat << "from component " << comp << " does not exist." << std::endl;
			return;
		}
		const std::vector <float>& data = bc->getData(feat);
		auto cur = dataPython.size();
		dataPython.push_back(QVector<double>());
		for (const auto val : data)
			dataPython[cur].push_back(val);
	}

	auto pos = filename.find_last_of(".");
	auto moduleName = filename.substr(0, pos);

	poca::core::PythonInterpreter* py = poca::core::PythonInterpreter::instance();
	QVector <double> res;
	bool res2 = py->applyFunctionWithNArraysParameterAnd1ArrayReturned(res, dataPython, moduleName.c_str(), nameFunction.c_str());
	if (res2 == EXIT_FAILURE)
		std::cout << "ERROR! Function << " << nameFunction << " from Python file " << filename << " was not run with error message : python script was not run." << std::endl;
	else {
		std::cout << "Result of running " << nameFunction << " from Python file " << filename << ":" << std::endl;
		std::copy(res.begin(), res.end(), std::ostream_iterator<double>(std::cout, " "));
		std::cout << std::endl;
	}
}

void PythonWidget::executePythonScriptAddFeatureToComponent(const poca::core::CommandInfo& _command)
{
	QVector <QVector <double>> dataPython;
	std::vector <std::string> features = _command.getParameter< std::vector <std::string>>("features");
	std::string filename = _command.getParameter<std::string>("filename");
	std::string nameFunction = _command.getParameter<std::string>("nameFunction");
	std::string component = _command.getParameter<std::string>("addToComponent");
	std::string nameFeature = _command.getParameter<std::string>("nameNewFeature");

	poca::core::MyObjectInterface* obj = m_object->currentObject();
	for (const auto& feature : features) {
		auto pos = feature.find(" -> ");
		std::string comp = feature.substr(0, pos), feat = feature.substr(pos + 4);
		//std::cout << comp << " ------ " << feat << std::endl;
		poca::core::BasicComponent* bc = obj->getBasicComponent(comp);
		if (!bc) {
			std::cout << "Error: execution of the Python function " << nameFunction << " from the file " << filename << " failed. Component " << comp << " does not exist." << std::endl;
			return;
		}
		if(!bc->hasData(feat)) {
			std::cout << "Error: execution of the Python function " << nameFunction << " from the file " << filename << " failed. Feature " << feat << "from component " << comp << " does not exist." << std::endl;
			return;
		}
		const std::vector <float>& data = bc->getData(feat);
		auto cur = dataPython.size();
		dataPython.push_back(QVector<double>());
		for (const auto val : data)
			dataPython[cur].push_back(val);
	}

	auto pos = filename.find_last_of(".");
	auto moduleName = filename.substr(0, pos);

	poca::core::PythonInterpreter* py = poca::core::PythonInterpreter::instance();
	QVector <double> res;
	bool res2 = py->applyFunctionWithNArraysParameterAnd1ArrayReturned(res, dataPython, moduleName.c_str(), nameFunction.c_str());
	if (res2 == EXIT_FAILURE)
		std::cout << "ERROR! Function << " << nameFunction << " from Python file " << filename <<" was not run with error message : python script was not run." << std::endl;
	else {
		std::vector <float> newFeature(res.size(), 0.f);
		std::transform(res.begin(), res.end(), newFeature.begin(), [](double x) { return (float)x; });
		poca::core::BasicComponent* bc = obj->getBasicComponent(component);
		if (!bc) {
			std::cout << "Error: execution of the Python function " << nameFunction << " from the file " << filename << " failed. Result should have been added to component " << component << " that does not exist." << std::endl;
			return;
		}
		bc->addFeature(nameFeature, new poca::core::MyData(newFeature));
		m_object->notify("LoadObjCharacteristicsAllWidgets");
	}
}

QStringList PythonWidget::identifyPythonFunctionNames(const QString& _file) const
{
	QStringList functions, splitFile;
	splitFile = _file.split("def ");
	for (auto n = 1; n < splitFile.size(); n++) {
		QString current = splitFile[n];
		auto pos = current.indexOf(":");
		if (pos == -1) continue;
		auto function = current.left(pos);
		functions.push_back(function);
	}
	return functions;
}

void PythonWidget::updateChosenFunctionName(int _idx)
{
	QString function = m_functionNameCombo->currentText();
	//Remove parameters
	auto pos = function.indexOf("(");
	if (pos != -1)
		function = function.left(pos);
	m_editNameFunction->setText(function);
}

#endif