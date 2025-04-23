/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MainWindow.cpp
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

#include <Windows.h>
#include <gl/glew.h>
#include <iostream>
#include <limits>
#include <random>
#include <cmath>
#include <fstream>
#include <QtCore/QSignalMapper>
#include <QtCore/QDir>
#include <QtCore/QPluginLoader>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QAction>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QMenu>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QApplication>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QLayout>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QActionGroup>
#include <QtGui/QDragEnterEvent>
#include <QtGui/QDropEvent>
#include <QtGui/QImage>
#include <QtCore/QMimeData>
#include <dtv.h>

#include <OpenGL/Camera.hpp>
#include <Geometry/DetectionSet.hpp>
#include <Interfaces/DelaunayTriangulationInterface.hpp>
#include <Interfaces/DelaunayTriangulationFactoryInterface.hpp>
#include <Interfaces/VoronoiDiagramFactoryInterface.hpp>
#include <Geometry/VoronoiDiagram.hpp>
#include <Plot/Icons.hpp>
#include <DesignPatterns/MediatorWObjectFWidget.hpp>
#include <General/BasicComponent.hpp>
#include <General/BasicComponentList.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/DelaunayTriangulationInterface.hpp>
#include <General/Misc.h>
#include <General/PluginList.hpp>
#include <General/PythonInterpreter.hpp>
#include <General/Engine.hpp>
#include <DesignPatterns/MacroRecorderSingleton.hpp>
#include <Objects/MyObjectDisplayCommand.hpp>
#include <OpenGL/Helper.h>
#include <General/MyData.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/ObjectIndicesFactoryInterface.hpp>
#include <General/Image.hpp>
#include <General/Engine.hpp>

#include "../../include/GuiInterface.hpp"
#include "../../include/PluginInterface.hpp"

#include "../Widgets/MainWindow.hpp"
#include "../Widgets/MdiChild.hpp"
#include "../Objects/SMLM_Object/SMLMObject.hpp"
#include "../Objects/Coloc_Object/ColocObject.hpp"
#include "../Widgets/MainFilterWidget.hpp"
#include "../Widgets/ColocalizationChoiceDialog.hpp"
#include "../Widgets/PythonWidget.hpp"
#include "../Widgets/ROIGeneralWidget.hpp"
#include "../Widgets/MacroWidget.hpp"
#include "../Widgets/PythonParametersDialog.hpp"

#undef max 

void decomposePathToDirAndFile(const QString& _path, QString& _dirQS, QString& _fileQS)
{
	int lasti = _path.lastIndexOf("/");
	if (lasti == -1) {
		QDir dir(_path);
		_dirQS = dir.absolutePath();
	}
	else {
		_fileQS = _path.mid(lasti + 1);
		if (_fileQS.contains(".")) {//We have a filename with an extension
			QDir dir(_path.mid(0, lasti));
			_dirQS = dir.absolutePath();
		}
		else {//No extension, then it is a repretory
			QDir dir(_path);
			_dirQS = dir.absolutePath();
			_fileQS.clear();
		}
	}
}

MainWindow::MainWindow() :m_firstLoad(true), m_currentDuplicate(1)
{
	// Get current directory
	poca::core::PrintFullPath(".\\");

	//Add needed path to environment variable PATH
	char buf[poca::core::ENV_BUF_SIZE];
	std::size_t bufsize = poca::core::ENV_BUF_SIZE;
	std::string pathToAdd = ".\\external\\";
	int e = getenv_s(&bufsize, buf, bufsize, "PATH");
	printf("value of PATH: %.*s\n", (int)sizeof(buf), buf);
	if (e) {
		//std::cerr << "`getenv_s` failed, returned " << e << '\n';
		//exit(EXIT_FAILURE);
	}
	std::string env_path, orig_path = buf;
	env_path = pathToAdd + ";";
	env_path += orig_path;
	e = _putenv_s("PATH", env_path.c_str());
	if (e) {
		std::cerr << "`_putenv_s` failed, returned " << e << std::endl;
	}
	std::cout << "new value of path: " << env_path << std::endl;

	QSurfaceFormat format;
	format.setVersion(2, 1);
	format.setProfile(QSurfaceFormat::CoreProfile);
	QSurfaceFormat::setDefaultFormat(format);

	poca::core::Engine* engine = poca::core::Engine::instance();
	engine->initialize();

	poca::core::MediatorWObjectFWidget* mediator = engine->getMediator();
	poca::core::MacroRecorderSingleton* macroRecord = std::any_cast <poca::core::MacroRecorderSingleton*>(engine->getSingleton("MacroRecorderSingleton"));
	m_mdiArea = new MyMdiArea;
	m_mdiArea->setObjectName("MdiArea");
	m_mdiArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	m_mdiArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	setCentralWidget(m_mdiArea);
	m_windowMapper = new QSignalMapper(this);

	createActions();
	createToolBars();
	createMenus();
	createStatusBar();

	m_tabWidget = new QTabWidget(this);
	m_tabWidget->setObjectName("TabWidget");
	m_tabWidget->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Expanding);
	m_tabWidget->setContentsMargins(0, 0, 0, 0);
	m_tabWidget->tabBar()->setMovable(true);
	m_mfw = new MainFilterWidget(mediator, m_tabWidget);
	mediator->addWidget(m_mfw);
	QTabWidget* tabMisc = new QTabWidget;
	tabMisc->addTab(m_mfw, QObject::tr("General"));
	m_tabWidget->addTab(tabMisc, QObject::tr("Misc."));
	QObject::connect(m_mfw, SIGNAL(savePosition(QString)), this, SLOT(savePositionCameraSlot(QString)));
	QObject::connect(m_mfw, SIGNAL(loadPosition(QString)), this, SLOT(loadPositionCameraSlot(QString)));
	QObject::connect(m_mfw, SIGNAL(pathCamera(QString, QString, float, bool, bool)), this, SLOT(pathCameraSlot(QString, QString, float, bool, bool)));
	QObject::connect(m_mfw, SIGNAL(pathCamera2(nlohmann::json, nlohmann::json, float, bool, bool)), this, SLOT(pathCameraSlot2(nlohmann::json, nlohmann::json, float, bool, bool)));
	QObject::connect(m_mfw, SIGNAL(pathCameraAll(const std::vector <std::tuple<float, glm::vec3, glm::quat>>&, bool, bool)), this, SLOT(pathCameraAllSlot(const std::vector <std::tuple<float, glm::vec3, glm::quat>>&, bool, bool)));
	QObject::connect(m_mfw, SIGNAL(getCurrentCamera()), this, SLOT(currentCameraForPath()));

	m_macroW = new MacroWidget(mediator, m_tabWidget);
	mediator->addWidget(m_macroW);
	m_tabWidget->addTab(m_macroW, QObject::tr("Macro"));
	macroRecord->setTextEdit(m_macroW->getTextEdit());
	macroRecord->setJson(m_macroW->getJson());
	m_macroW->loadParameters(engine->getGlobalParameters());

	engine->addGUI(m_tabWidget);

#ifndef NO_PYTHON
	m_pythonW = new PythonWidget(mediator, m_tabWidget);
	mediator->addWidget(m_pythonW);
	m_tabWidget->addTab(m_pythonW, QObject::tr("Python"));
	m_pythonW->loadParameters(engine->getGlobalParameters());
#endif

	m_ROIsW = new ROIGeneralWidget(mediator, m_tabWidget);
	mediator->addWidget(m_ROIsW);
	poca::core::utils::addWidget(m_tabWidget, QString("ROI Manager"), QString("General"), m_ROIsW, false);

	for (int n = 0; n < m_tabWidget->count(); n++) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
		m_tabWidget->setTabVisible(n, m_tabWidget->tabText(n) == "Misc." || m_tabWidget->tabText(n) == "Macro");
#endif
		QTabWidget* tab = dynamic_cast <QTabWidget*>(m_tabWidget->widget(n));
		if (!tab) continue;
		std::string name = m_tabWidget->tabText(n).toStdString();
		std::string name2 = tab->tabText(0).toStdString();
		int cur = tab->currentIndex();
		std::string name3 = tab->tabText(cur).toStdString();
		tab->setCurrentIndex(0);
		int cur2 = tab->currentIndex();
		std::string name4 = tab->tabText(cur2).toStdString();
		cur2++;
	}

	m_tabWidget->setCurrentWidget(m_macroW);

	QHBoxLayout* layoutColor = new QHBoxLayout;
	layoutColor->setContentsMargins(0, 0, 0, 0);
	layoutColor->setSpacing(0);
	QWidget* emptyWleft = new QWidget;
	emptyWleft->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QWidget* emptyWright = new QWidget;
	emptyWright->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	m_widgetColors = new QWidget;
	m_widgetColors->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	layoutColor->addWidget(emptyWleft);
	layoutColor->addWidget(m_widgetColors);
	layoutColor->addWidget(emptyWright);
	QWidget* colorW = new QWidget;
	colorW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	colorW->setLayout(layoutColor);
	m_colorButtonsGroup = new QButtonGroup;
	QObject::connect(m_colorButtonsGroup, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(changeColorObject(QAbstractButton*)));

	QVBoxLayout* layoutAll = new QVBoxLayout;
	layoutAll->setContentsMargins(0, 0, 0, 0);
	layoutAll->setSpacing(0);
	layoutAll->addWidget(colorW);
	layoutAll->addWidget(m_tabWidget);
	QWidget* widgetAll = new QWidget;
	widgetAll->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	widgetAll->setLayout(layoutAll);

	QDockWidget* dock = new QDockWidget(this);
	dock->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
	dock->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	//
	QScrollArea* area = new QScrollArea;
	area->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	area->setWidgetResizable(true);
	dock->setWidget(area);
	area->setWidget(widgetAll);
	dock->setWindowTitle(tr(""));
	addDockWidget(Qt::RightDockWidgetArea, dock);

	area->resize(600, area->height());

	setActiveMdiChild(NULL);

	setWindowTitle(tr("PoCA: Point Cloud Analyst"));
	setUnifiedTitleAndToolBarOnMac(true);
	statusBar()->showMessage(tr("Ready"));
	m_lblPermanentStatus = new QLabel;
	m_lblPermanentStatus->setVisible(false);
	m_progressBar = new QProgressBar(this);
	m_progressBar->setVisible(false);
	statusBar()->addPermanentWidget(m_lblPermanentStatus);
	statusBar()->addPermanentWidget(m_progressBar);

	poca::core::NbObjects = m_mfw->getFirstIndexObj();

	setAcceptDrops(true);

	QObject::connect(m_macroW, SIGNAL(runMacro(std::vector<nlohmann::json>)), this, SLOT(runMacro(std::vector<nlohmann::json>)));
	QObject::connect(m_macroW, SIGNAL(runMacro(std::vector<nlohmann::json>, QStringList)), this, SLOT(runMacro(std::vector<nlohmann::json>, QStringList)));
}

MainWindow::~MainWindow()
{
	nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();

	if (m_currentMdi != NULL) {
		m_currentMdi->getWidget()->getObject()->saveCommands(parameters);
	}
	poca::core::CommandInfo command(false, "saveParameters", "file", &parameters);
	poca::core::Engine* engine = poca::core::Engine::instance();
	engine->getPlugins()->execute(&command);
	m_macroW->execute(&command);
	m_pythonW->execute(&command);

	std::string text = parameters.dump(), textDisplay = parameters.dump(4);
	std::cout << textDisplay << std::endl;
	std::ofstream ofs("poca.ini");
	ofs << text;
	ofs.close();
}

void MainWindow::closeEvent(QCloseEvent * event)
{
}

void MainWindow::createActions()
{
	m_openFileAct = new QAction(QIcon(QPixmap(poca::plot::openFileIcon)), tr("&Open file"), this);
	m_openFileAct->setShortcuts(QKeySequence::Open);
	m_openFileAct->setStatusTip(tr("Open an existing file"));
	QObject::connect(m_openFileAct, SIGNAL(triggered()), this, SLOT(openFile()));

	m_openDirAct = new QAction(QIcon(QPixmap(poca::plot::openDirIcon)), tr("&Open directory"), this);
	m_openDirAct->setStatusTip(tr("Open an existing directory"));
	QObject::connect(m_openDirAct, SIGNAL(triggered()), this, SLOT(openDir()));

	m_plusAct = new QAction(QIcon(QPixmap(poca::plot::plusIcon)), tr("&Add component"), this);
	m_plusAct->setStatusTip(tr("Add component to current dataset"));
	QObject::connect(m_plusAct, SIGNAL(triggered()), this, SLOT(addComponentToCurrentMdi()));

	m_duplicateAct = new QAction(QIcon("./images/duplicate.png"), tr("Duplicate localizations"), this);
	m_duplicateAct->setStatusTip(tr("Open an existing file"));
	QObject::connect(m_duplicateAct, SIGNAL(triggered()), this, SLOT(duplicate()));
	
	m_closeAllAct = new QAction(QIcon("./images/closeAllIcon.png"), tr("&Close All"), this);
	m_closeAllAct->setShortcuts(QKeySequence::Close);
	m_closeAllAct->setStatusTip(tr("Close all datasets"));
	QObject::connect(m_closeAllAct, SIGNAL(triggered()), this, SLOT(closeAllDatasets()));

	m_tileWindowsAct = new QAction(QIcon("./images/tileWindows.png"), tr("&Tile Windows"), this);
	m_tileWindowsAct->setStatusTip(tr("Tile Windows"));
	QObject::connect(m_tileWindowsAct, SIGNAL(triggered()), this, SLOT(tileWindows()));
	m_cascadeWindowsAct = new QAction(QIcon("./images/cascadeWindow.png"), tr("&Cascade Windows"), this);
	m_cascadeWindowsAct->setStatusTip(tr("Cascade Windows"));
	QObject::connect(m_cascadeWindowsAct, SIGNAL(triggered()), this, SLOT(cascadeWindows()));

	m_aboutAct = new QAction(QIcon("./images/about.png"), tr("About..."), this);
	m_aboutAct->setStatusTip(tr("About..."));
	QObject::connect(m_aboutAct, SIGNAL(triggered()), this, SLOT(aboutDialog()));

	m_exitAct = new QAction(tr("E&xit"), this);
	m_exitAct->setShortcuts(QKeySequence::Quit);
	m_exitAct->setStatusTip(tr("Exit the application"));
	connect(m_exitAct, SIGNAL(triggered()), qApp, SLOT(closeAllWindows()));

	m_cropAct = new QAction(QIcon("./images/crop.png"), tr("&Crop"), this);
	m_cropAct->setCheckable(true);
	m_cropAct->setChecked(false);
	m_cropAct->setEnabled(false);
	m_cropAct->setStatusTip(tr("Crop dataset"));
	QObject::connect(m_cropAct, SIGNAL(toggled(bool)), this, SLOT(setCameraInteraction(bool)));
	m_xyAct = new QAction(QIcon(QPixmap(poca::plot::xyIcon)), tr("&XY plane"), this);
	m_xyAct->setCheckable(true);
	m_xyAct->setChecked(false);
	connect(m_xyAct, SIGNAL(toggled(bool)), this, SLOT(setCameraInteraction(bool)));
	connect(m_xyAct, SIGNAL(triggered()), this, SLOT(setCameraInteraction()));
	m_xzAct = new QAction(QIcon(QPixmap(poca::plot::xzIcon)), tr("&XZ plane"), this);
	m_xzAct->setCheckable(true);
	m_xzAct->setChecked(false);
	connect(m_xzAct, SIGNAL(toggled(bool)), this, SLOT(setCameraInteraction(bool)));
	m_yzAct = new QAction(QIcon(QPixmap(poca::plot::yzIcon)), tr("&YZ plane"), this);
	m_yzAct->setCheckable(true);
	m_yzAct->setChecked(false);
	connect(m_yzAct, SIGNAL(toggled(bool)), this, SLOT(setCameraInteraction(bool)));
	QActionGroup* actGroup = new QActionGroup(this);
	actGroup->addAction(m_xyAct);
	actGroup->addAction(m_xzAct);
	actGroup->addAction(m_yzAct);
	actGroup->setExclusionPolicy(QActionGroup::ExclusionPolicy::ExclusiveOptional);

	m_resetProjAct = new QAction(QIcon(QPixmap(poca::plot::resetProjIcon)), tr("&Reset viewer"), this);
	m_resetProjAct->setStatusTip(tr("Reset viewer"));
	connect(m_resetProjAct, SIGNAL(triggered()), this, SLOT(resetViewer()));

	m_boundingBoxAct = new QAction(QIcon("./images/boundingBox.png"), tr("Toggle bounding box"), this);
	m_boundingBoxAct->setStatusTip(tr("Toggle bounding box"));
	m_boundingBoxAct->setCheckable(true);
	m_boundingBoxAct->setChecked(true);
	QObject::connect(m_boundingBoxAct, SIGNAL(triggered()), this, SLOT(toggleBoundingBoxDisplay()));

	m_gridAct = new QAction(QIcon("./images/grid.png"), tr("&Toggle Grid"), this);
	m_gridAct->setStatusTip(tr("Toggle Grid"));
	m_gridAct->setCheckable(true);
	m_gridAct->setChecked(true);
	QObject::connect(m_gridAct, SIGNAL(triggered()), this, SLOT(toggleGridDisplay()));

	m_fontDisplayAct = new QAction(QIcon(QPixmap(poca::plot::fontDisplayIcon)), tr("&Toggle Fonts"), this);
	m_fontDisplayAct->setStatusTip(tr("Toggle Fonts"));
	m_fontDisplayAct->setCheckable(true);
	m_fontDisplayAct->setChecked(true);
	QObject::connect(m_fontDisplayAct, SIGNAL(triggered()), this, SLOT(toggleFontDisplay()));

	m_colocAct = new QAction(QIcon("./images/colocalization.png"), tr("Colocalization"), this);
	m_colocAct->setStatusTip(tr("Colocalization"));
	QObject::connect(m_colocAct, SIGNAL(triggered()), this, SLOT(computeColocalization()));

	//ROIs
	m_line2DROIAct = new QAction(QIcon(QPixmap(poca::plot::line2DIcon)), tr("&Line 2D"), this);
	m_line2DROIAct->setStatusTip(tr("Line 2D ROI"));
	m_line2DROIAct->setCheckable(true);
	connect(m_line2DROIAct, SIGNAL(triggered()), this, SLOT(setCameraInteraction()));

	m_triangle2DROIAct = new QAction(QIcon(QPixmap(poca::plot::triangle2DIcon)), tr("&Triangle 2D"), this);
	m_triangle2DROIAct->setStatusTip(tr("Triangle 2D ROI"));
	m_triangle2DROIAct->setCheckable(true);
	connect(m_triangle2DROIAct, SIGNAL(triggered()), this, SLOT(setCameraInteraction()));

	m_circle2DROIAct = new QAction(QIcon(QPixmap(poca::plot::circle2DIcon)), tr("&Circle 2D"), this);
	m_circle2DROIAct->setStatusTip(tr("Circle 2D ROI"));
	m_circle2DROIAct->setCheckable(true);
	connect(m_circle2DROIAct, SIGNAL(triggered()), this, SLOT(setCameraInteraction()));

	m_square2DROIAct = new QAction(QIcon(QPixmap(poca::plot::square2DIcon)), tr("&Square 2D"), this);
	m_square2DROIAct->setStatusTip(tr("Square 2D ROI"));
	m_square2DROIAct->setCheckable(true);
	connect(m_square2DROIAct, SIGNAL(triggered()), this, SLOT(setCameraInteraction()));

	m_polyline2DROIAct = new QAction(QIcon(QPixmap(poca::plot::polyline2DIcon)), tr("&Polyline 2D"), this);
	m_polyline2DROIAct->setStatusTip(tr("Polyline 2D ROI"));
	m_polyline2DROIAct->setCheckable(true);
	connect(m_polyline2DROIAct, SIGNAL(triggered()), this, SLOT(setCameraInteraction()));

	m_sphere3DROIAct = new QAction(QIcon(QPixmap(poca::plot::sphere3DIcon)), tr("&Sphere 3D"), this);
	m_sphere3DROIAct->setStatusTip(tr("Sphere 3D ROI"));
	m_sphere3DROIAct->setCheckable(true);
	connect(m_sphere3DROIAct, SIGNAL(triggered()), this, SLOT(setCameraInteraction()));

	m_planeROIAct = new QAction(QIcon(QPixmap(poca::plot::planeROIIcon)), tr("&Plane"), this);
	m_planeROIAct->setStatusTip(tr("Plane ROI"));
	m_planeROIAct->setCheckable(true);
	connect(m_planeROIAct, SIGNAL(triggered()), this, SLOT(setCameraInteraction()));

	m_polyplaneROIAct = new QAction(QIcon(QPixmap(poca::plot::planeROIIcon)), tr("&PolyPlane"), this);
	m_polyplaneROIAct->setStatusTip(tr("PolyPlane ROI"));
	m_polyplaneROIAct->setCheckable(true);
	connect(m_polyplaneROIAct, SIGNAL(triggered()), this, SLOT(setCameraInteraction()));

	QActionGroup* roiGroup = new QActionGroup(this);
	roiGroup->setExclusionPolicy(QActionGroup::ExclusionPolicy::ExclusiveOptional);
	roiGroup->addAction(m_line2DROIAct);
	roiGroup->addAction(m_triangle2DROIAct);
	roiGroup->addAction(m_circle2DROIAct);
	roiGroup->addAction(m_square2DROIAct);
	roiGroup->addAction(m_polyline2DROIAct);
	roiGroup->addAction(m_sphere3DROIAct);
	roiGroup->addAction(m_planeROIAct);
	roiGroup->addAction(m_polyplaneROIAct);
}

void MainWindow::createMenus()
{
	QMenuBar* menuB = menuBar();

	QMenu* fileMenu = menuB->addMenu("File");
	QMenu* openMenu = fileMenu->addMenu("Open");
	openMenu->addAction(m_openFileAct);
	openMenu->addAction(m_openDirAct);
	openMenu->addAction(m_plusAct);

	poca::core::Engine* engine = poca::core::Engine::instance();
	const std::vector <PluginInterface*>& plugins = engine->getPlugins()->getPlugins();
	for (size_t n = 0; n < plugins.size(); n++) {
		std::vector <std::pair<QAction*, QString>> actions = plugins[n]->getActions();
		for (std::pair<QAction*, QString> action : actions) {
			QList <QAction*> globalActions = menuB->actions();

			if (action.second.startsWith("Toolbar")) continue;

			QStringList menus = action.second.split("/");

			//First determine if a new entry to the menu bar has to be added
			QAction* act = NULL;
			QList <QAction*> actions = globalActions;
			for (QAction* cur : globalActions) {
				if (cur->text() == menus[0])
					act = cur;
			}

			if (act == NULL) {
				QMenu* cur = menuB->addMenu(menus[0]);
				for (int n = 1; n < menus.size(); n++)
					cur = cur->addMenu(menus[n]);
				cur->addAction(action.first);
				connect(action.first, SIGNAL(triggered()), this, SLOT(actionFromPlugin()));
			}
			else {
				QMenu* cur = act->menu();
				for (int n = 1; n < menus.size(); n++) {
					QList <QAction*> actions = cur->actions();
					QAction* found = NULL;
					for (QAction* curAction : actions) {
						if (curAction->text() == menus[n])
							found = curAction;
					}
					if (found)
						cur = found->menu();
					else
						cur = cur->addMenu(menus[n]);
				}
				cur->addAction(action.first);
				connect(action.first, SIGNAL(triggered()), this, SLOT(actionFromPlugin()));
			}
		}
	}

#ifndef NO_PYTHON
	QList <QAction*> globalActions = menuB->actions();
	QAction* act = NULL;
	for (QAction* cur : globalActions) {
		if (cur->text() == "Plugins")
			act = cur;
	}
	QMenu* cur = NULL;
	if (act == NULL) 
		cur = menuB->addMenu("Plugins");
	else 
		cur = act->menu();
	cur = cur->addMenu("Python");
	m_pythonParamsAct = new QAction(tr("Parameters"), this);
	cur->addAction(m_pythonParamsAct);
	connect(m_pythonParamsAct, SIGNAL(triggered()), this, SLOT(setParametersPython()));
#endif
}

void MainWindow::setParametersPython()
{
	
	
	nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();

	PythonParametersDialog* ppd = new PythonParametersDialog(parameters);
	ppd->setModal(true);
	if (ppd->exec() == QDialog::Accepted) {
		std::cout << "Here" << std::endl;
		const std::vector <std::string>& names = ppd->getNameParameters();
		const std::vector <std::string>& paths = ppd->getPaths();
		for (auto n = 0; n < paths.size(); n++) {
			if (paths[n].empty()) continue;
			parameters["PythonParameters"][names[n]] = paths[n];
		}
		std::string textDisplay = parameters.dump(4);
		std::cout << textDisplay << std::endl;
	}
	delete ppd;
}

void MainWindow::createToolBars()
{
	m_fileToolBar = new QToolBar(tr("Toolbar"));
	m_fileToolBar->addAction(m_openFileAct);
	m_fileToolBar->addAction(m_openDirAct);
	m_fileToolBar->addAction(m_plusAct);
	m_fileToolBar->addAction(m_duplicateAct);
	m_fileToolBar->addSeparator();
	m_lastActionQuantifToolbar = m_fileToolBar->addSeparator();
	m_fileToolBar->addAction(m_line2DROIAct);
	m_fileToolBar->addAction(m_triangle2DROIAct);
	m_fileToolBar->addAction(m_circle2DROIAct);
	m_fileToolBar->addAction(m_square2DROIAct);
	m_fileToolBar->addAction(m_polyline2DROIAct);
	m_fileToolBar->addAction(m_sphere3DROIAct);
	m_fileToolBar->addAction(m_planeROIAct);
	m_fileToolBar->addAction(m_polyplaneROIAct);
	m_lastActionROIToolbar = m_fileToolBar->addSeparator();
	m_fileToolBar->addAction(m_colocAct);
	m_lastActionColocToolbar = m_fileToolBar->addSeparator();
	m_fileToolBar->addAction(m_cropAct);
	m_fileToolBar->addAction(m_xyAct);
	m_fileToolBar->addAction(m_xzAct);
	m_fileToolBar->addAction(m_yzAct);
	m_fileToolBar->addAction(m_resetProjAct);
	m_fileToolBar->addAction(m_boundingBoxAct);
	m_fileToolBar->addAction(m_gridAct);
	m_fileToolBar->addAction(m_fontDisplayAct);
	m_lastActionDisplayToolbar = m_fileToolBar->addSeparator();
	m_fileToolBar->addAction(m_tileWindowsAct);
	m_fileToolBar->addAction(m_cascadeWindowsAct);
	m_lastActionMiscToolbar = m_fileToolBar->addSeparator();
	m_fileToolBar->addAction(m_closeAllAct); 
	m_fileToolBar->addAction(m_aboutAct);

	poca::core::Engine* engine = poca::core::Engine::instance();
	const std::vector <PluginInterface*>& plugins = engine->getPlugins()->getPlugins();
	for (size_t n = 0; n < plugins.size(); n++) {
		std::vector <std::pair<QAction*, QString>> actions = plugins[n]->getActions();
		for (std::pair<QAction*, QString> action : actions) {
			if (!action.second.startsWith("Toolbar")) continue;
			QString val = action.second.right(action.second.size() - (action.second.indexOf("/") + 1));
			if(val == "1Color")
				m_fileToolBar->insertAction(m_lastActionQuantifToolbar, action.first);
			else if (val == "2Color")
				m_fileToolBar->insertAction(m_lastActionColocToolbar, action.first);
			else if (val == "Display")
				m_fileToolBar->insertAction(m_lastActionDisplayToolbar, action.first);
			else if (val == "Misc")
				m_fileToolBar->insertAction(m_lastActionMiscToolbar, action.first);
			else if (val == "SeparatorLast") {
				m_fileToolBar->addSeparator();
				m_fileToolBar->addAction(action.first);
			}
			else
				m_fileToolBar->addAction(action.first);
			connect(action.first, SIGNAL(triggered()), this, SLOT(actionFromPlugin()));

		}
	}

	addToolBar(Qt::LeftToolBarArea, m_fileToolBar);
}

void MainWindow::setCameraInteraction()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	size_t dimension = cam->getObject()->dimension();
	QObject* sender = QObject::sender();
	if (sender == m_xyAct) {
		if (dimension == 3)
			return;
		else {
			cam->fixPlane(poca::opengl::Camera::Plane_XY, true);
			cam->fixPlane(poca::opengl::Camera::Plane_XY, false);
			return;
		}
	}
	if (sender == m_line2DROIAct)
		cam->setCameraInteraction(m_line2DROIAct->isChecked() ? poca::opengl::Camera::Line2DRoiDefinition : poca::opengl::Camera::None);
	else if (sender == m_triangle2DROIAct)
		cam->setCameraInteraction(m_triangle2DROIAct->isChecked() ? poca::opengl::Camera::Triangle2DRoiDefinition : poca::opengl::Camera::None);
	else if (sender == m_circle2DROIAct)
		cam->setCameraInteraction(m_circle2DROIAct->isChecked() ? poca::opengl::Camera::Circle2DRoiDefinition : poca::opengl::Camera::None);
	else if (sender == m_square2DROIAct)
		cam->setCameraInteraction(m_square2DROIAct->isChecked() ? poca::opengl::Camera::Square2DRoiDefinition : poca::opengl::Camera::None);
	else if (sender == m_polyline2DROIAct)
		cam->setCameraInteraction(m_polyline2DROIAct->isChecked() ? poca::opengl::Camera::Polyline2DRoiDefinition : poca::opengl::Camera::None);
	else if (sender == m_sphere3DROIAct)
		cam->setCameraInteraction(m_sphere3DROIAct->isChecked() ? poca::opengl::Camera::Sphere3DRoiDefinition : poca::opengl::Camera::None);
	else if (sender == m_planeROIAct)
		cam->setCameraInteraction(m_planeROIAct->isChecked() ? poca::opengl::Camera::PlaneRoiDefinition : poca::opengl::Camera::None);
	else if (sender == m_polyplaneROIAct)
		cam->setCameraInteraction(m_polyplaneROIAct->isChecked() ? poca::opengl::Camera::PolyPlaneRoiDefinition : poca::opengl::Camera::None);
}

void MainWindow::setCameraInteraction(bool _on)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	size_t dimension = cam->getObject()->dimension();
	QObject* sender = QObject::sender();
	QAction* act = NULL;
	if (sender == m_xyAct) {
		cam->fixPlane(poca::opengl::Camera::Plane_XY, _on);
		act = m_xyAct;
	}
	else if (sender == m_xzAct) {
		cam->fixPlane(poca::opengl::Camera::Plane_XZ, _on);
		act = m_xzAct;
	}
	else if (sender == m_yzAct) {
		cam->fixPlane(poca::opengl::Camera::Plane_YZ, _on);
		act = m_yzAct;
	}
	else if (sender == m_cropAct) {
		cam->setCameraInteraction(_on ? poca::opengl::Camera::Crop : poca::opengl::Camera::None);

		size_t dimension = cam->getObject()->dimension();
		m_line2DROIAct->setEnabled(!_on || dimension == 2);
		m_triangle2DROIAct->setEnabled(!_on || dimension == 2);
		m_circle2DROIAct->setEnabled(!_on || dimension == 2);
		m_square2DROIAct->setEnabled(!_on || dimension == 2);
		m_polyline2DROIAct->setEnabled(!_on || dimension == 2);
	}

	if (dimension == 2) return;
	if (act == m_xyAct) {
		m_line2DROIAct->setEnabled(_on);
		m_triangle2DROIAct->setEnabled(_on);
		m_circle2DROIAct->setEnabled(_on);
		m_square2DROIAct->setEnabled(_on);
		m_polyline2DROIAct->setEnabled(_on);
		if (!_on) {
			bool ROI2D = false;
			ROI2D |= m_line2DROIAct->isChecked();
			ROI2D |= m_triangle2DROIAct->isChecked();
			ROI2D |= m_circle2DROIAct->isChecked();
			ROI2D |= m_square2DROIAct->isChecked();
			ROI2D |= m_polyline2DROIAct->isChecked();

			m_line2DROIAct->setChecked(false);
			m_triangle2DROIAct->setChecked(false);
			m_circle2DROIAct->setChecked(false);
			m_square2DROIAct->setChecked(false);
			m_polyline2DROIAct->setChecked(false);

			if (ROI2D)
				cam->setCameraInteraction(poca::opengl::Camera::None);
		}
	}
	else {
		m_line2DROIAct->setEnabled(false);
		m_triangle2DROIAct->setEnabled(false);
		m_circle2DROIAct->setEnabled(false);
		m_square2DROIAct->setEnabled(false);
		m_polyline2DROIAct->setEnabled(false);
		m_line2DROIAct->setChecked(false);
		m_triangle2DROIAct->setChecked(false);
		m_circle2DROIAct->setChecked(false);
		m_square2DROIAct->setChecked(false);
		m_polyline2DROIAct->setChecked(false);
	}

	if (act != NULL)
		m_cropAct->setEnabled(act->isChecked());
	if(!m_xyAct->isChecked() && !m_xzAct->isChecked() && !m_yzAct->isChecked())// && cam->getCameraInteraction() == poca::opengl::Camera::Crop)
		m_cropAct->setChecked(false);
}

void MainWindow::actionFromPlugin()
{
	poca::core::MyObjectInterface* obj = NULL;
	if (m_currentMdi != NULL) {
		poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
		if(cam != NULL)
			obj = cam->getObject();
	}

	QObject* sender = QObject::sender();
	poca::core::Engine* engine = poca::core::Engine::instance();
	obj = engine->getPlugins()->actionTriggered(sender, obj);

	if (obj != NULL) {
		createWidget(obj);
	}
	setActiveMdiChild(m_currentMdi);
}


void MainWindow::createStatusBar()
{
	statusBar()->showMessage(tr("Ready"));
}

void MainWindow::keyPressEvent(QKeyEvent * _e)
{
}

void MainWindow::dragEnterEvent(QDragEnterEvent* _e)
{
	if (_e->mimeData()->hasUrls())
		_e->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent* _e)
{
	auto urls = _e->mimeData()->urls();
	for (auto url : urls) {
		QString name = url.toLocalFile();
		execute(&poca::core::CommandInfo(true, "open", "path", name.toStdString()));
		if (name.endsWith(".txt")) {
			poca::core::CommandInfo ci(true, "openFile", "name", name.toStdString());
			poca::core::Engine* engine = poca::core::Engine::instance();
			engine->getPlugins()->execute(&ci);
			if (!ci.hasParameter("object")) continue;
			poca::core::MyObjectInterface* obj = ci.getParameterPtr<poca::core::MyObjectInterface>("object");
			if (obj != NULL) {
				createWidget(obj);
			}
		}
	}
}

void MainWindow::createObjectFromFeatures(const std::map <std::string, std::vector <float>>& _features, const std::string _dir, const std::string _name)
{
	poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(_features);
	poca::core::MyObject* wobj = new poca::core::MyObject();
	wobj->setDir(_dir.c_str());
	wobj->setName(_name.c_str());
	wobj->addBasicComponent(dset);
	wobj->setDimension(dset->dimension());
	createWidget(wobj);
}

void MainWindow::createWidget(poca::core::MyObjectInterface* _obj)
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::PluginList* plugins = engine->getPlugins();
	poca::opengl::Camera* cam = new poca::opengl::Camera(_obj, _obj->dimension(), NULL);// this);

	poca::core::MediatorWObjectFWidget* mediator = poca::core::MediatorWObjectFWidget::instance();
	poca::core::SubjectInterface* subject = dynamic_cast<poca::core::SubjectInterface*>(_obj);
	if (subject) {
		mediator->addObserversToSubject(subject, "LoadObjCharacteristicsAllWidgets");
		mediator->addObserversToSubject(subject, "UpdateMainTabWidgets");
		mediator->addObserversToSubject(subject, "LoadObjCharacteristicsMiscWidget");
		mediator->addObserversToSubject(subject, "LoadObjCharacteristicsDetectionSetWidget");
		mediator->addObserversToSubject(subject, "LoadObjCharacteristicsDelaunayTriangulationWidget");
		mediator->addObserversToSubject(subject, "LoadObjCharacteristicsVoronoiDiagramWidget");
	}
	_obj->attach(cam, "updateDisplay");
	_obj->attach(cam, "updateInfosObject");
	_obj->attach(this, "addCommandLastAddedComponent");
	_obj->attach(this, "addCommandToSpecificComponent");
	_obj->attach(this, "LoadObjCharacteristicsAllWidgets");
	_obj->attach(this, "UpdateMainTabWidgets");
	_obj->attach(this, "duplicateCleanedData");

	_obj->addCommand(new MyObjectDisplayCommand(_obj));
	for (auto& bci : _obj->getComponents()) {
		//poca::core::BasicComponentInterface* bci = _obj->getLastAddedBasicComponent();
		if (bci != NULL && bci->nbCommands() == 0)
			plugins->addCommands(bci);

		if (bci != NULL) {
			poca::core::BasicComponentList* blist = dynamic_cast<poca::core::BasicComponentList*>(bci);
			if (blist)
				for (auto bcomp : blist->components())
					if (bcomp->nbCommands() == 0)
						plugins->addCommands(bcomp);
		}
	}

	MdiChild* child = new MdiChild(cam);
	QObject::connect(child, SIGNAL(setCurrentMdi(MdiChild*)), this, SLOT(setActiveMdiChild(MdiChild*)));
	QObject::connect(cam, SIGNAL(askForMovieCreation()), this, SLOT(createMovie()));
	m_mdiArea->addSubWindow(child);
	setActiveMdiChild(child);
	child->layout()->update();
	child->layout()->activate();
	child->getWidget()->update();
	child->show();

	_obj->notify("LoadObjCharacteristicsAllWidgets");
	_obj->notifyAll("updateDisplay");

	engine->addData(_obj, cam);
}


void MainWindow::openFile()
{
	QString path = QDir::currentPath();
	QStringList filenames = QFileDialog::getOpenFileNames(0,
		QObject::tr("Select one or more files to open"),
		path,
		QObject::tr("Localization files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

	if (filenames.isEmpty()) return;

	for (const QString& filename : filenames)
		execute(&poca::core::CommandInfo(true, "open", "path", std::string(filename.toStdString())));
}

void MainWindow::openDir()
{
	QString path = QDir::currentPath();
	QString dirName = QFileDialog::getExistingDirectory(0,
		QObject::tr("Select directory"),
		path,
		QFileDialog::DontUseNativeDialog | QFileDialog::DontResolveSymlinks);

	if (dirName.isEmpty()) return;

	QDir dir(dirName);
	dir.setFilter(QDir::Files | QDir::NoSymLinks);

	if (!dirName.endsWith("/"))
		dirName.append("/");

	QFileInfoList list = dir.entryInfoList();
	for (int i = 0; i < list.size(); ++i) {
		QFileInfo fileInfo = list.at(i);
		QString filename = fileInfo.fileName();
		if (!filename.endsWith(".csv")) continue;
		execute(&poca::core::CommandInfo(true, "open", "path", (dirName + filename).toStdString()));
	}
}

void MainWindow::addComponentToCurrentMdi()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	poca::core::MyObjectInterface* obj = cam->getObject();
	if (obj == NULL) return;

	QString path = QDir::currentPath();
	QString filename = QFileDialog::getOpenFileName(0,
		QObject::tr("Select one component to add"),
		path,
		QObject::tr("Component files (*.*)"), 0, QFileDialog::DontUseNativeDialog);

	if (filename.isEmpty()) return;

	poca::core::CommandInfo ci(false, "open", "path", std::string(filename.toStdString()));

	poca::core::Engine* engine = poca::core::Engine::instance();
	if (engine->loadDataAndAddToObject(filename, obj, &ci)) {
		obj->notify("LoadObjCharacteristicsAllWidgets");
		obj->notifyAll("updateDisplay");
	}
}

void MainWindow::openFile(const QString& _filename, poca::core::CommandInfo* _command)
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::PluginList* plugins = engine->getPlugins();
	poca::core::MyObjectInterface* obj = engine->loadDataAndCreateObject(_filename, _command);
	if (obj == NULL)
		return;
	poca::opengl::CameraInterface* cam = createWindows(obj);
	engine->addCameraToObject(obj, cam);
}

void MainWindow::duplicate()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	poca::core::MyObjectInterface* obj = cam->getObject();
	if (obj == NULL) return;

	update(dynamic_cast <poca::core::SubjectInterface*>(obj), "duplicateOrganoidCentroids");

	poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
	poca::core::BasicComponentInterface* bci = obj->getBasicComponent("DetectionSet");
	poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
	if (dset == NULL) return;
	poca::geometry::DetectionSet* newDset = dset->duplicateSelection();
	const std::string& dir = obj->getDir(), name = obj->getName();
	QString newName(name.c_str());
	int index = newName.lastIndexOf(".");
	newName.insert(index, QString("_%1").arg(m_currentDuplicate++));
	createWindows(newDset, QString(dir.c_str()), newName);
}

poca::opengl::CameraInterface* MainWindow::createWindows(poca::core::MyObjectInterface* _obj)
{
	if (_obj != NULL) {
		poca::core::MyObject* obj = static_cast<poca::core::MyObject*>(_obj);
		if (obj == NULL)
			return NULL;
		poca::opengl::Camera* cam = new poca::opengl::Camera(_obj, _obj->dimension(), NULL);// this);

		int indexVoronoiTab = 0;

		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MediatorWObjectFWidget* mediator = engine->getMediator();// poca::core::MediatorWObjectFWidget::instance();

		mediator->addObserversToSubject(obj, "LoadObjCharacteristicsAllWidgets");
		mediator->addObserversToSubject(obj, "UpdateMainTabWidgets");
		mediator->addObserversToSubject(obj, "LoadObjCharacteristicsMiscWidget");
		mediator->addObserversToSubject(obj, "LoadObjCharacteristicsDetectionSetWidget");
		mediator->addObserversToSubject(obj, "LoadObjCharacteristicsDelaunayTriangulationWidget");
		mediator->addObserversToSubject(obj, "LoadObjCharacteristicsVoronoiDiagramWidget");
		obj->attach(cam, "updateDisplay");
		obj->attach(cam, "updateInfosObject");

		obj->attach(this, "addCommandLastAddedComponent");
		obj->attach(this, "addCommandToSpecificComponent");
		obj->attach(this, "LoadObjCharacteristicsAllWidgets");
		obj->attach(this, "UpdateMainTabWidgets");
		obj->attach(this, "duplicateCleanedData");

		MdiChild* child = new MdiChild(cam);
		QObject::connect(child, SIGNAL(setCurrentMdi(MdiChild*)), this, SLOT(setActiveMdiChild(MdiChild*)));
		QObject::connect(cam, SIGNAL(askForMovieCreation()), this, SLOT(createMovie()));
		m_mdiArea->addSubWindow(child);
		setActiveMdiChild(child);

		child->layout()->update();
		child->layout()->activate();
		child->getWidget()->update();
		child->show();

		return static_cast <poca::opengl::CameraInterface*>(cam);
	}
	return NULL;
}

poca::core::MyObjectInterface* MainWindow::createWindows(poca::core::BasicComponent* _bc, const QString& _dir, const QString& _name)
{
	if (_bc == NULL) return NULL;

	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::PluginList* plugins = engine->getPlugins();

	poca::core::MyObject* wobj = new SMLMObject();
	wobj->setDir(_dir.toLatin1().data());
	wobj->setName(_name.toLatin1().data());
	wobj->addBasicComponent(_bc);
	wobj->setDimension(_bc->dimension());

	if (wobj != NULL) {
		poca::opengl::Camera* cam = new poca::opengl::Camera(wobj, _bc->dimension(), NULL);// this);

		int indexVoronoiTab = 0;

		poca::core::MediatorWObjectFWidget* mediator = engine->getMediator();// poca::core::MediatorWObjectFWidget::instance();

		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsAllWidgets");
		mediator->addObserversToSubject(wobj, "UpdateMainTabWidgets");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsMiscWidget");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsDetectionSetWidget");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsDelaunayTriangulationWidget");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsVoronoiDiagramWidget");
		wobj->attach(cam, "updateDisplay");
		wobj->attach(cam, "updateInfosObject");

		wobj->attach(this, "addCommandLastAddedComponent");
		wobj->attach(this, "addCommandToSpecificComponent");
		wobj->attach(this, "LoadObjCharacteristicsAllWidgets");	
		wobj->attach(this, "UpdateMainTabWidgets");
		wobj->attach(this, "duplicateCleanedData");
		
		SMLMObject* sobj = dynamic_cast <SMLMObject*>(wobj);
		sobj->addCommand(new MyObjectDisplayCommand(sobj)); 

		plugins->addCommands(_bc);
		plugins->addCommands(wobj);

		MdiChild* child = new MdiChild(cam);
		QObject::connect(child, SIGNAL(setCurrentMdi(MdiChild*)), this, SLOT(setActiveMdiChild(MdiChild*)));
		QObject::connect(cam, SIGNAL(askForMovieCreation()), this, SLOT(createMovie()));
		m_mdiArea->addSubWindow(child);
		setActiveMdiChild(child);

		child->layout()->update();
		child->layout()->activate();
		child->getWidget()->update();
		child->show();

		//poca::geometry::DelaunayTriangulationFactoryInterface* factory = poca::geometry::createDelaunayTriangulationFactory();
		//poca::geometry::DelaunayTriangulationInterface* delaunay = factory->createDelaunayTriangulation(wobj, NULL, false);
		//delete factory;
	}
	return wobj;
}

void MainWindow::setActiveMdiChild(MdiChild * _mdiChild)
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::MediatorWObjectFWidget* mediator = engine->getMediator();//poca::core::MediatorWObjectFWidget::instance();
	if (_mdiChild == NULL) {
		m_currentMdi = NULL;
		if (m_mdiArea->subWindowList().isEmpty())
			mediator->setCurrentObject(NULL);
		else {
			QMdiSubWindow* window = m_mdiArea->subWindowList().front();
			_mdiChild = qobject_cast <MdiChild*>(window);
		}
	}
	if (_mdiChild && _mdiChild != m_currentMdi){
		poca::core::MyObjectInterface * wobj = _mdiChild->getWidget()->getObject();
		mediator->setCurrentObject(wobj);
		wobj->notify("LoadObjCharacteristicsAllWidgets");
		m_currentMdi = _mdiChild;

		int maxSize = 20;
		QHBoxLayout* layout = NULL;
		size_t nbColors = wobj->nbColors();
		if (m_colorButtons.empty()) {
			layout = new QHBoxLayout;
			layout->setContentsMargins(0, 0, 0, 0);
			layout->setSpacing(0);
			for (size_t n = 0; n < nbColors; n++) {
				QPushButton* button = new QPushButton(QString::number(n+1));
				button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
				button->setMaximumSize(QSize(maxSize, maxSize));
				button->setCheckable(true);
				m_colorButtons.push_back(button);
				layout->addWidget(button, 0, Qt::AlignCenter);
				m_colorButtonsGroup->addButton(button, n);
			}
			m_widgetColors->setLayout(layout);
		}
		else if (nbColors > m_colorButtons.size()) {
			layout = dynamic_cast <QHBoxLayout*>(m_widgetColors->layout());
			//Here, we need to add some hist widgets because this loc data has more features than the one loaded before
			for (size_t n = m_colorButtons.size(); n < nbColors; n++) {
				QPushButton* button = new QPushButton(QString::number(n + 1));
				button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
				button->setMaximumSize(QSize(maxSize, maxSize));
				button->setCheckable(true);
				m_colorButtons.push_back(button);
				layout->addWidget(button, 0, Qt::AlignCenter);
				m_colorButtonsGroup->addButton(button, n);
			}
		}
		else if (nbColors <= m_colorButtons.size()) {
			//Here, wee have less feature to display than hist widgets available, we hide the ones that are unecessary
			for (size_t n = 0; n < m_colorButtons.size(); n++)
				m_colorButtons[n]->setVisible(n < nbColors);
		}
		m_colorButtonsGroup->button(wobj->currentObjectID())->setChecked(true);
		m_widgetColors->updateGeometry();

		size_t dimension = wobj->dimension();
		m_line2DROIAct->setEnabled(dimension == 2);
		m_triangle2DROIAct->setEnabled(dimension == 2);
		m_circle2DROIAct->setEnabled(dimension == 2);
		m_square2DROIAct->setEnabled(dimension == 2);
		m_polyline2DROIAct->setEnabled(dimension == 2);
		m_sphere3DROIAct->setEnabled(dimension == 3);
		m_planeROIAct->setEnabled(dimension == 3);
		m_polyplaneROIAct->setEnabled(dimension == 3);

		if (m_xyAct->isChecked())
			m_xyAct->setChecked(false);
		if (m_xzAct->isChecked())
			m_xzAct->setChecked(false);
		if (m_yzAct->isChecked())
			m_yzAct->setChecked(false);
		
		m_xyAct->setCheckable(dimension == 3);
		m_xzAct->setEnabled(dimension == 3);
		m_yzAct->setEnabled(dimension == 3);

		poca::opengl::Camera* cam = dynamic_cast<poca::opengl::Camera*>(_mdiChild->getWidget());
		if (cam != NULL) {
			cam->setCameraInteraction(poca::opengl::Camera::None);
			cam->fixPlane(poca::opengl::Camera::None, false);
		}

		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(wobj);
		if (!comObj) return;

		if (comObj->hasParameter("fontDisplay")) {
			bool displayFont = comObj->getParameter<bool>("fontDisplay");
			m_fontDisplayAct->blockSignals(true);
			m_fontDisplayAct->setChecked(displayFont);
			m_fontDisplayAct->blockSignals(false);
		}
	}
	updateTabWidget();
}

void MainWindow::aboutDialog()
{
	QString text("POCA is developed by Florian Levet (florian.levet@inserm.fr),\n");
	text.append("research engineer in the Quantitative Imaging of the Cell team,\n");
	text.append("directed by Jean-Baptiste Sibarita.\n");
	text.append("F.L. and J.B.S. are part of the Interdisciplinary Institute for Neuroscience.\n");
	text.append("http://www.iins.u-bordeaux.fr/\n");
	text.append("F.L. is part of the Bordeaux Imaging Center.\n");
	text.append("http://www.bic.u-bordeaux.fr/");
	QMessageBox message(QMessageBox::NoIcon, "About...", text);
	message.setIconPixmap(QPixmap("./images/voronIcon1_2.PNG"));
	message.setWindowIcon(QIcon("./images/voronIcon1.PNG"));
	message.setMinimumWidth(1200);
	message.exec();
}

void MainWindow::closeAllDatasets()
{
	m_mdiArea->closeAllSubWindows();
}

MdiChild * MainWindow::getChild(const unsigned int _idx)
{
	foreach(QMdiSubWindow * window, m_mdiArea->subWindowList()) {
		MdiChild * mdiChild = qobject_cast <MdiChild *>(window);
		poca::core::MyObjectInterface * obj = mdiChild->getWidget()->getObject();
		if (obj->currentInternalId() == _idx) return mdiChild;
	}
	return NULL;
}

QWidget * MainWindow::getFilterWidget(const QString & _name)
{
	QWidget * widget = NULL;
	for (int n = 0; n < m_tabWidget->count() && widget == NULL; n++){
		QWidget * tmp = m_tabWidget->widget(n);
		if (tmp->objectName() == _name) widget = tmp;
	}
	return widget;
}

void MainWindow::setPermanentStatusText(const QString & _text){
	m_lblPermanentStatus->setText(_text);
	m_lblPermanentStatus->setVisible(!_text.isEmpty());
}

void MainWindow::tileWindows()
{
	m_mdiArea->tileSubWindows();
}

void MainWindow::cascadeWindows()
{
	m_mdiArea->cascadeSubWindows();
}

void MainWindow::update(poca::core::SubjectInterface* _subj, const poca::core::CommandInfo& _action)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*>(_subj);
	if (obj == NULL) return;

	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::PluginList* plugins = engine->getPlugins();

	if (_action == "LoadObjCharacteristicsAllWidgets" || _action == "UpdateMainTabWidgets")
		updateTabWidget();
	if(_action == "addCommandToSpecificComponent"){
		poca::core::BasicComponentInterface* comp = _action.getParameterPtr<poca::core::BasicComponentInterface>("component");
		plugins->addCommands(comp);
		poca::core::Engine::instance()->getObject(obj)->notify("LoadObjCharacteristicsAllWidgets");
		updateTabWidget();
		obj->notify("LoadObjCharacteristicsAllWidgets");
	}
	if (_action == "addCommandLastAddedComponent") {
		poca::core::BasicComponentInterface* bci = obj->getLastAddedBasicComponent();
		if (bci == NULL) return;
		plugins->addCommands(bci);

		poca::core::Engine::instance()->getObject(obj)->notify("LoadObjCharacteristicsAllWidgets");

		updateTabWidget();
		obj->notify("LoadObjCharacteristicsAllWidgets");
	}
	if (_action == "duplicateCleanedData") {
		poca::core::CommandInfo ci(false, "getCleanedData");
		obj->executeCommandOnSpecificComponent("DetectionSet", &ci);
		if (ci.hasParameter("detectionSet")) {
			poca::geometry::DetectionSet* dset = ci.getParameterPtr <poca::geometry::DetectionSet>("detectionSet");
			const std::string& dir = obj->getDir(), name = obj->getName();
			QString newName(name.c_str());
			int index = newName.lastIndexOf(".");
			newName.insert(index, QString("_%1").arg(m_currentDuplicate++));
			createWindows(dset, QString(dir.c_str()), newName);
		}
	}
}

void MainWindow::resetViewer()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	cam->resetProjection();
	cam->update();
}

void MainWindow::toggleBoundingBoxDisplay()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	cam->toggleBoundingBoxDisplay();
	cam->repaint();
}

void MainWindow::toggleGridDisplay()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	cam->toggleGridDisplay();
	cam->repaint();
}

void MainWindow::toggleFontDisplay()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	cam->toggleFontDisplay();
	cam->repaint();
}

void MainWindow::computeColocalization()
{
	std::vector < std::pair < QString, MdiChild* > > datasets;
	foreach(QMdiSubWindow * window, m_mdiArea->subWindowList()) {
		MdiChild* mdiChild = qobject_cast <MdiChild*>(window);
		poca::core::MyObjectInterface* sobj = mdiChild->getWidget()->getObject();
		QString dir = sobj->getDir().c_str(), name = sobj->getName().c_str(), completeName;
		if (dir.endsWith("/")) dir = dir.mid(0, dir.size() - 1);
		completeName = dir;
		if (dir.split("/").back() != name)
			completeName.append("/").append(name);
		datasets.push_back(std::make_pair(completeName, mdiChild));// sobj->currentInternalId()));
	}
	if (datasets.size() < 2) {
		std::cout << "At least two datasets with Voronoi diagram are needed for colocalization analysis" << std::endl;
		return;
	}

	ColocalizationChoiceDialog* dial = new ColocalizationChoiceDialog(datasets);
	dial->setModal(true);
	if (dial->exec() == QDialog::Accepted) {
		std::vector < MdiChild*> objects = dial->getObjects();

		computeColocalization(objects);
	}
	delete dial;
}

void MainWindow::computeColocalization(const int _id0, const int _id1)
{
	MdiChild* ws[2] = { NULL, NULL };
	poca::core::MyObjectInterface* obj1 = NULL, * obj2 = NULL;
	foreach(QMdiSubWindow * window, m_mdiArea->subWindowList()) {
		MdiChild* mdiChild = qobject_cast <MdiChild*>(window);
		poca::core::MyObjectInterface* obj = mdiChild->getWidget()->getObject();
		if (obj == NULL) continue;
		if (ws[0] == NULL)
			ws[0] = mdiChild;
		else if (ws[1] == NULL)
			ws[1] = mdiChild;
	}
	std::vector < MdiChild*> objects = { ws[0], ws[1] };
	if (ws[0] != NULL && ws[1] != NULL)
		computeColocalization(objects);
}

void MainWindow::computeColocalization(const std::vector < std::string>& _nameDatasets)
{
	std::vector < MdiChild*> ws;
	QList<QMdiSubWindow*> widgets = m_mdiArea->subWindowList();
	for (const std::string& name : _nameDatasets) {
		MdiChild* w = NULL;
		for (QList<QMdiSubWindow*>::const_iterator it = widgets.begin(); it != widgets.end() && w == NULL; it++) {
			MdiChild* mdiChild = qobject_cast <MdiChild*>(*it);
			poca::core::MyObjectInterface* obj = mdiChild->getWidget()->getObject();
			std::string currentName = obj->getName();
			if (name == currentName)
				w = mdiChild;
		}
		if (w != NULL)
			ws.push_back(w);
	}
	if (ws.size() < 2) {
		std::cout << "PoCA did not manage to create a colocalization dataset with the names ";
		for (const std::string& name : _nameDatasets)
			std::cout << name << ", ";
		std::cout << std::endl;
	}
	else
		computeColocalization(ws);
}

void MainWindow::computeColocalization(const std::vector < MdiChild*>& _ws)
{
	std::vector<poca::core::MyObjectInterface*> objs;
	for (MdiChild* mc : _ws)
		objs.push_back(mc->getWidget()->getObject());

	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::PluginList* plugins = engine->getPlugins();

	poca::core::MyObjectInterface* wobj = engine->generateMultipleObject(objs);
	if (wobj == NULL) return;

	if (wobj != NULL) {
		std::vector <std::string> names;
		for (MdiChild* mdi : _ws)
			names.push_back(mdi->getWidget()->getObject()->getName());
		poca::core::MacroRecorderSingleton::instance()->addCommand("MainWindow", &poca::core::CommandInfo(true, "computeColocalization", "datasetNames", names));

		for (MdiChild* mc : _ws) {
			poca::opengl::CameraInterface* camW = mc->getWidget();
			camW->makeCurrent();
			m_mdiArea->removeSubWindow(mc);
			delete mc;
		}

		poca::opengl::Camera* cam = new poca::opengl::Camera(wobj, wobj->dimension(), this);
		engine->addCameraToObject(wobj, cam);

		int indexVoronoiTab = 0;

		poca::core::MediatorWObjectFWidget* mediator = poca::core::MediatorWObjectFWidget::instance();

		poca::core::SubjectInterface* si = dynamic_cast<poca::core::SubjectInterface*>(wobj);

		mediator->addObserversToSubject(si, "LoadObjCharacteristicsAllWidgets");
		mediator->addObserversToSubject(si, "LoadObjCharacteristicsMiscWidget");
		mediator->addObserversToSubject(si, "LoadObjCharacteristicsDetectionSetWidget");
		mediator->addObserversToSubject(si, "LoadObjCharacteristicsDelaunayTriangulationWidget");
		mediator->addObserversToSubject(si, "LoadObjCharacteristicsVoronoiDiagramWidget");
		wobj->attach(cam, "updateDisplay");
		wobj->attach(cam, "updateInfosObject");
		wobj->attach(cam, "updateInfosObjectOverlap");

		wobj->attach(this, "addCommandLastAddedComponent");

		wobj->addCommand(new MyObjectDisplayCommand(wobj));
		
		for (size_t n = 0; n < objs.size(); n++) {
			objs[n]->attach(cam, "updateDisplay");
			objs[n]->attach(this, "addCommandLastAddedComponent");
			objs[n]->attach(this, "addCommandToSpecificComponent");
			objs[n]->attach(this, "LoadObjCharacteristicsAllWidgets");
		}

		MdiChild* child = new MdiChild(cam);
		QObject::connect(child, SIGNAL(setCurrentMdi(MdiChild*)), this, SLOT(setActiveMdiChild(MdiChild*)));
		m_mdiArea->addSubWindow(child);
		setActiveMdiChild(child);

		child->layout()->update();
		child->layout()->activate();
		child->getWidget()->update();
		child->show();

		wobj->notify("LoadObjCharacteristicsAllWidgets");
		updateTabWidget();
	}
}

void MainWindow::changeColorObject(QAbstractButton* _button)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	poca::core::MyObjectInterface* obj = cam->getObject();
	if (obj == NULL) return;

	int index = m_colorButtonsGroup->id(_button);
	obj->setCurrentObject(index);
	obj->notify("LoadObjCharacteristicsAllWidgets");
	obj->notifyAll("updateDisplay");
}

void MainWindow::currentCameraForPath()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	m_mfw->setCurrentCamera(cam);
}

void MainWindow::savePositionCameraSlot(QString _filename)
{
	execute(&poca::core::CommandInfo(false, "savePositionCamera"));
}

void MainWindow::savePositionCamera()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;

	QString filename("cameraPosition.json");
	filename = QFileDialog::getSaveFileName(NULL, QObject::tr("Save camera position..."), filename, QString("json files (*.json)"), 0, QFileDialog::DontUseNativeDialog);
	if (filename.isEmpty()) return;
	execute(&poca::core::CommandInfo(true, "savePositionCamera", "path", filename.toStdString()));
}

void MainWindow::savePositionCamera(const std::string& _filename)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;

	QString filename(_filename.c_str());
	if (filename.isEmpty()) return;
	if (!filename.endsWith(".json"))
		filename.append(".json");

	const poca::opengl::StateCamera& stateCam = cam->getStateCamera();
	nlohmann::json json;
	json["stateCamera"]["matrixView"] = stateCam.m_matrixView;
	json["stateCamera"]["rotationSum"] = stateCam.m_rotationSum;
	json["stateCamera"]["rotation"] = stateCam.m_rotation;
	json["stateCamera"]["center"] = stateCam.m_center;
	json["stateCamera"]["eye"] = stateCam.m_eye;
	json["stateCamera"]["matrix"] = stateCam.m_matrix;
	json["stateCamera"]["up"] = stateCam.m_up;
	json["stateCamera"]["translationModel"] = cam->getTranslationModel();
	json["distanceOrtho"] = cam->getDistanceOrtho();
	json["distanceOrthoOriginal"] = cam->getOriginalDistanceOrtho();
	json["crop"] = cam->getCurrentCrop();

	std::string text = json.dump();
	std::cout << text << std::endl;
	std::ofstream fs(filename.toLatin1().data());
	fs << text;
	fs.close();
}

void MainWindow::loadPositionCameraSlot(QString _filename)
{
	if(_filename.isEmpty())
		execute(&poca::core::CommandInfo(false, "loadPositionCamera"));
	else
		execute(&poca::core::CommandInfo(true, "loadPositionCamera", "path", _filename.toStdString()));
}

void MainWindow::pathCameraSlot(QString _pos1, QString _pos2, float _duration, bool _saveImages, bool _traveling)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;

	std::array <QString, 2> names = { _pos1, _pos2 };
	std::array <poca::opengl::StateCamera, 2> scams;
	std::array <float, 2> distances;

	for (auto n = 0; n < 2; n++) {
		nlohmann::json json;
		std::ifstream fs(names[n].toStdString());
		if (fs.good())
			fs >> json;
		fs.close();

		if (json.contains("stateCamera")) {
			nlohmann::json tmp = json["stateCamera"];
			if (tmp.contains("matrixView"))
				scams[n].m_matrixView = tmp["matrixView"].get<glm::mat4>();
			if (tmp.contains("rotationSum"))
				scams[n].m_rotationSum = tmp["rotationSum"].get<glm::quat>();
			if (tmp.contains("rotation"))
				scams[n].m_rotation = tmp["rotation"].get<glm::quat>();
			if (tmp.contains("center"))
				scams[n].m_center = tmp["center"].get<glm::vec3>();
			if (tmp.contains("eye"))
				scams[n].m_eye = tmp["eye"].get<glm::vec3>();
			if (tmp.contains("up"))
				scams[n].m_up = tmp["up"].get<glm::vec3>();
			if (tmp.contains("translationModel"))
				scams[n].m_translationModel = tmp["translationModel"].get<glm::vec3>();
		}
		if (json.contains("distanceOrtho"))
			distances[n] = json["distanceOrtho"].get<float>();
	}

	cam->animateCameraPath(scams, distances, _duration, _saveImages, _traveling);
}

void MainWindow::pathCameraSlot2(nlohmann::json _pos1, nlohmann::json _pos2, float _duration, bool _saveImages, bool _traveling)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;

	std::array <nlohmann::json, 2> jsons = { _pos1, _pos2 };
	std::array <poca::opengl::StateCamera, 2> scams;
	std::array <float, 2> distances;

	for (auto n = 0; n < 2; n++) {
		const nlohmann::json& json = jsons[n];
		if (json.contains("stateCamera")) {
			nlohmann::json tmp = json["stateCamera"];
			if (tmp.contains("matrixView"))
				scams[n].m_matrixView = tmp["matrixView"].get<glm::mat4>();
			if (tmp.contains("rotationSum"))
				scams[n].m_rotationSum = tmp["rotationSum"].get<glm::quat>();
			if (tmp.contains("rotation"))
				scams[n].m_rotation = tmp["rotation"].get<glm::quat>();
			if (tmp.contains("center"))
				scams[n].m_center = tmp["center"].get<glm::vec3>();
			if (tmp.contains("eye"))
				scams[n].m_eye = tmp["eye"].get<glm::vec3>();
			if (tmp.contains("up"))
				scams[n].m_up = tmp["up"].get<glm::vec3>();
			if (tmp.contains("translationModel"))
				scams[n].m_translationModel = tmp["translationModel"].get<glm::vec3>();
		}
		if (json.contains("distanceOrtho"))
			distances[n] = json["distanceOrtho"].get<float>();
	}

	cam->animateCameraPath(scams, distances, _duration, _saveImages, _traveling);
}

void MainWindow::pathCameraAllSlot(const std::vector <std::tuple<float, glm::vec3, glm::quat>>& _iterations, bool _saveImages, bool _traveling)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	cam->animateCameraPath(_iterations, _saveImages, _traveling);
}

void MainWindow::createMovie()
{
	QObject* obj = this->sender();
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(obj);
	if (cam == NULL)
		return;

	QString filename("movie.mp4");
	filename = QFileDialog::getSaveFileName(NULL, QObject::tr("Save movie..."), filename, QString("mp4 files (*.mp4)"), 0, QFileDialog::DontUseNativeDialog);
	if (filename.isEmpty()) return;

	const std::vector <QImage>& _frames = cam->getMovieFrames();

	atg_dtv::Encoder encoder;
	atg_dtv::Encoder::VideoSettings settings{};

	// Output filename
	settings.fname = filename.toStdString();

	// Input dimensions
	settings.inputWidth = _frames[0].width();
	settings.inputHeight = _frames[0].height();

	// Output dimensions
	settings.width = _frames[0].width();
	settings.height = _frames[0].height();

	// Encoder settings
	settings.hardwareEncoding = true;
	settings.bitRate = 16000000;
	settings.frameRate = 24;

	const int FrameCount = _frames.size();

	auto start = std::chrono::steady_clock::now();

	std::cout << "==============================================\n";
	std::cout << " Direct to Video (DTV) Sample Application\n\n";

	encoder.run(settings, 2);

	for (int i = 0; i < FrameCount; ++i) {
		if ((i + 1) % 100 == 0 || i >= FrameCount - 10) {
			std::cout << "Frame: " << (i + 1) << "/" << FrameCount << "\n";
		}

		const int sin_i = std::lroundf(255 * (0.5 + 0.5 * std::sin(i * 0.01)));

		atg_dtv::Frame* frame = encoder.newFrame(true);
		if (frame == nullptr) break;
		if (encoder.getError() != atg_dtv::Encoder::Error::None) break;

		const int lineWidth = frame->m_lineWidth;
		for (int y = 0; y < settings.inputHeight; ++y) {
			uint8_t* row = &frame->m_rgb[y * lineWidth];
			for (int x = 0; x < settings.inputWidth; ++x) {
				const int index = x * 3;
				QRgb color = _frames[i].pixel(x, y);
				row[index + 0] = qRed(color); // r
				row[index + 1] = qGreen(color); // g
				row[index + 2] = qBlue(color);   // b
			}
		}

		/*QString paddedNumber = QString::number(i).rightJustified(5, '0');
		bool res = _frames[i].save(QString("e:/poca_") + paddedNumber + QString(".png"));
		if (!res)
			std::cout << "Problem with saving" << std::endl;*/

		encoder.submitFrame();

		bool res = _frames[i].save(QString("d:/poca_%1.jpg").arg(QString::number(i + 1).rightJustified(3, '0')));
		if (!res)
			std::cout << "Problem with saving" << std::endl;
	}

	encoder.commit();
	encoder.stop();

	auto end = std::chrono::steady_clock::now();

	const double elapsedSeconds =
		std::chrono::duration<double>(end - start).count();

	std::cout << "==============================================\n";
	if (encoder.getError() == atg_dtv::Encoder::Error::None) {
		std::cout << "Encoding took: " << elapsedSeconds << " seconds" << "\n";
		std::cout << "Real-time framerate: " << FrameCount / elapsedSeconds << " FPS" << "\n";
	}
	else {
		std::cout << "Encoding failed\n";
	}
}

void MainWindow::loadPositionCamera()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;

	QString filename;
	QString path = QDir::currentPath();
	filename = QFileDialog::getOpenFileName(0,
		QObject::tr("Camera position"),
		path,
		QObject::tr("Camera position (*.json)"), 0, QFileDialog::DontUseNativeDialog);

	if (filename.isEmpty()) return;
	bool view = m_mfw->isViewCameraChecked(); 
	bool rotation = m_mfw->isRotationCameraChecked();
	bool translation = m_mfw->isTranslationCameraChecked();
	bool zoom = m_mfw->isZoomCameraChecked();
	bool crop = m_mfw->isCropCameraChecked();
	execute(&poca::core::CommandInfo(true, "loadPositionCamera", "path", filename.toStdString(), "view", view, "rotation", rotation, "translation", translation, "zoom", zoom, "crop", crop));
}

void MainWindow::loadPositionCamera(const std::string& _filename, const bool _reset, const bool _view, const bool _rotation, const bool _translation, const bool _zoom, const bool _crop)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;

	if (_filename.empty()) return;
	nlohmann::json json;
	std::ifstream fs(_filename);
	if (fs.good())
		fs >> json;
	fs.close();

	if (json.contains("stateCamera")) {
		poca::opengl::StateCamera& stateCam = cam->getStateCamera();
		nlohmann::json tmp = json["stateCamera"];
		if (tmp.contains("matrixView") && _view)
			stateCam.m_matrixView = tmp["matrixView"].get<glm::mat4>();
		if (tmp.contains("rotationSum") && _rotation)
			stateCam.m_rotationSum = tmp["rotationSum"].get<glm::quat>();
		if (tmp.contains("rotation") && _rotation)
			stateCam.m_rotation = tmp["rotation"].get<glm::quat>();
		if (tmp.contains("center") && _view)
			stateCam.m_center = tmp["center"].get<glm::vec3>();
		if (tmp.contains("eye") && _view)
			stateCam.m_eye = tmp["eye"].get<glm::vec3>();
		if (tmp.contains("up") && _view)
			stateCam.m_up = tmp["up"].get<glm::vec3>();
		if (tmp.contains("translationModel") && _translation)
			stateCam.m_translationModel = tmp["translationModel"].get<glm::vec3>();

	}
	if (json.contains("distanceOrtho") && _zoom)
		cam->setDistanceOrtho(json["distanceOrtho"].get<float>());
	if (json.contains("crop") && _crop)
		cam->setCurrentCrop(json["crop"].get<poca::core::BoundingBox>());

	cam->zoomToBoundingBox(cam->getCurrentCrop(), _reset);
	cam->getObject()->notifyAll("updateDisplay");
}

void MainWindow::zoomToCropCurrentMdi(poca::core::BoundingBox _crop)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	cam->zoomToBoundingBox(_crop);
	cam->getObject()->notifyAll("updateDisplay");
}

void MainWindow::updateTabWidget()
{
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	for (int n = 0; n < m_tabWidget->count(); n++) {
		QTabWidget* tab = dynamic_cast <QTabWidget*>(m_tabWidget->widget(n));
		if (!tab) continue;
		bool oneTabVisible = false;
		for (int j = 0; j < tab->count(); j++)
			oneTabVisible = oneTabVisible || tab->isTabVisible(j);
		m_tabWidget->setTabVisible(n, oneTabVisible);
	}
	if (m_currentMdi == NULL) {
		for (int n = 0; n < m_tabWidget->count(); n++)
			m_tabWidget->setTabVisible(n, m_tabWidget->tabText(n) == "Misc." || m_tabWidget->tabText(n) == "Macro");
	}
#endif
}

void MainWindow::execute(poca::core::CommandInfo* _com)
{
	if (_com->nameCommand == "open") {
		std::string filename;
		if (_com->hasParameter("path"))
			filename = _com->getParameter<std::string>("path");
		if (filename.empty())
			openFile();
		else {
			QFileInfo info(filename.c_str());
			if(!info.exists()){
				QMessageBox msgBox;
				msgBox.setText("The file " + QString(filename.c_str()) + " does not exist");
				msgBox.exec();
				return;
			}
			openFile(info.absoluteFilePath(), _com);
		}
	}
	else if (_com->nameCommand == "close") {
		QList<QMdiSubWindow*> windows = m_mdiArea->subWindowList();
		if (m_currentMdi != NULL) {
			m_currentMdi->close();
			windows = m_mdiArea->subWindowList();
			m_currentMdi = windows.empty() ? NULL : static_cast <MdiChild*>(windows[windows.size() - 1]);
		}
	}
	else if (_com->nameCommand == "computeColocalization") {
		std::vector <std::string> names = _com->getParameter<std::vector <std::string>>("datasetNames");
		computeColocalization(names);
	}
	else if (_com->nameCommand == "changeViewerSize") {
		if (m_currentMdi != NULL && _com->hasParameter("width") && _com->hasParameter("height")) {
			uint32_t w = _com->getParameter<uint32_t>("width");
			uint32_t h = _com->getParameter<uint32_t>("height");
			m_currentMdi->resize(w, h);
		}
	}
	else if (_com->nameCommand == "savePositionCamera") {
		if (_com->hasParameter("path")) {
			std::string filename = _com->getParameter<std::string>("path");
			savePositionCamera(filename);
		}
		else
			savePositionCamera();
	}
	else if (_com->nameCommand == "loadPositionCamera") {
		bool reset = _com->hasParameter("reset") ? _com->getParameter<bool>("reset") : false;
		bool view = _com->hasParameter("view") ? _com->getParameter<bool>("view") : true;
		bool rotation = _com->hasParameter("rotation") ? _com->getParameter<bool>("rotation") : true;
		bool translation = _com->hasParameter("translation") ? _com->getParameter<bool>("translation") : true;
		bool zoom = _com->hasParameter("zoom") ? _com->getParameter<bool>("zoom") : true;
		bool crop = _com->hasParameter("crop") ? _com->getParameter<bool>("crop") : true;
		if (_com->hasParameter("path")) {
			std::string filename = _com->getParameter<std::string>("path");
			std::cout << "reset " << reset << std::endl;
			loadPositionCamera(filename, reset, view, rotation, translation, zoom, crop);
		}
		else
			loadPositionCamera();
	}
	if (_com->isRecordable())
		poca::core::MacroRecorderSingleton::instance()->addCommand("MainWindow", _com);
}

void MainWindow::actionNeeded()
{
}

void MainWindow::runMacro(std::vector<nlohmann::json> _macro)
{
	for (auto json : _macro) {
		if (json.empty()) continue;

		const auto nameComp = json.begin().key();
		if (nameComp == "MainWindow")
			runMacro(json[nameComp]);
		else if (nameComp == "PythonWidget") {
			nlohmann::json jsonCommand = json[nameComp];
			for (auto& [nameCommand, value] : jsonCommand.items()) {
				nlohmann::json parameters;
				poca::core::CommandInfo command = m_pythonW->createCommand(nameCommand, jsonCommand[nameCommand]);
				if (!command.empty())
					m_pythonW->execute(&command);
				else
					std::cout << "Widget [" << nameComp << "], command [" << nameCommand << "] does not exist, command " << jsonCommand.dump() << " was not executed." << std::endl;
			}
		}
		else {
			if (m_currentMdi == NULL) continue;
			poca::core::CommandableObject* comObj = NULL;
			poca::core::MyObjectInterface* obj = m_currentMdi->getWidget()->getObject();
			if (nameComp == "Object")
				comObj = dynamic_cast<poca::core::CommandableObject*>(obj);
			else {
				comObj = dynamic_cast<poca::core::CommandableObject*>(obj->getBasicComponent(nameComp));
			}
			if (comObj != NULL) {
				nlohmann::json jsonCommand = json[nameComp];
				for (auto& [nameCommand, value] : jsonCommand.items()) {
					nlohmann::json parameters;
					poca::core::CommandInfo command = comObj->createCommand(nameCommand, jsonCommand[nameCommand]);
					if (!command.empty()) {
						comObj->executeCommand(&command);
						if (command.hasParameter("newObject")) {
							poca::geometry::DetectionSet* dset = command.getParameterPtr<poca::geometry::DetectionSet>("newObject");
							if (dset == NULL) return;
							poca::geometry::DetectionSet* newDset = dset->duplicateSelection();
							const std::string& dir = obj->getDir(), name = obj->getName();
							QString newName(name.c_str());
							int index = newName.lastIndexOf(".");
							newName.insert(index, QString("_%1").arg(m_currentDuplicate++));
							createWindows(newDset, QString(dir.c_str()), newName);
						}
						else if (command.hasParameter("object")) {
							poca::core::MyObjectInterface* obj = command.getParameterPtr<poca::core::MyObjectInterface>("object");
							createWidget(obj);
						}
					}
					else
						std::cout << "Component [" << nameComp << "], command [" << nameCommand << "] does not exist, command " << jsonCommand.dump() << " was not executed." << std::endl;
				}
			}
			else
				std::cout << "Component [" << nameComp << "] does not exist, command " << json.dump() << " was not executed." << std::endl;
		}

	}
}

void MainWindow::runMacro(std::vector<nlohmann::json> _macro, QStringList _filenames)
{
	for (auto filename : _filenames) {
		for (auto json : _macro) {
			if (json.empty()) continue;

			const auto nameComp = json.begin().key();
			if (nameComp == "MainWindow") {
				auto command = json[nameComp];
				if (command.contains("open")) {
					//if (command["open"].contains("path"))
						command["open"]["path"] = filename.toStdString();
				}
				runMacro(command);
			}
			else if (nameComp == "PythonWidget") {
				nlohmann::json jsonCommand = json[nameComp];
				for (auto& [nameCommand, value] : jsonCommand.items()) {
					nlohmann::json parameters;
					poca::core::CommandInfo command = m_pythonW->createCommand(nameCommand, jsonCommand[nameCommand]);
					if (!command.empty())
						m_pythonW->execute(&command);
					else
						std::cout << "Widget [" << nameComp << "], command [" << nameCommand << "] does not exist, command " << jsonCommand.dump() << " was not executed." << std::endl;
				}
			}
			else {
				if (m_currentMdi == NULL) continue;
				poca::core::CommandableObject* comObj = NULL;
				if (nameComp == "Object")
					comObj = dynamic_cast<poca::core::CommandableObject*>(m_currentMdi->getWidget()->getObject());
				else {
					poca::core::MyObjectInterface* obj = m_currentMdi->getWidget()->getObject();
					comObj = dynamic_cast<poca::core::CommandableObject*>(obj->getBasicComponent(nameComp));
				}
				if (comObj != NULL) {
					nlohmann::json jsonCommand = json[nameComp];
					for (auto& [nameCommand, value] : jsonCommand.items()) {
						nlohmann::json parameters;
						poca::core::CommandInfo command = comObj->createCommand(nameCommand, jsonCommand[nameCommand]);
						if (!command.empty()) {
							comObj->executeCommand(&command);
							if (command.hasParameter("object")) {
								poca::core::MyObjectInterface* obj = command.getParameterPtr<poca::core::MyObjectInterface>("object");
								createWidget(obj);
							}
						}
						else
							std::cout << "Component [" << nameComp << "], command [" << nameCommand << "] does not exist, command " << jsonCommand.dump() << " was not executed." << std::endl;
					}
				}
				else
					std::cout << "Component [" << nameComp << "] does not exist, command " << json.dump() << " was not executed." << std::endl;
			}

		}
	}
}

void MainWindow::runMacro(const nlohmann::json& _json)
{
	if (_json.empty()) return;
	const auto tmp = _json.begin().key();
	if (tmp == "open") {
		poca::core::CommandInfo command(false, tmp);

		for (auto& [key, value] : _json[tmp].items()) {
			if (key == "path")
				command.addParameter(key, _json[tmp][key].get<std::string>());
			else if (key == "calibration_xy")
				command.addParameter(key, _json[tmp][key].get<float>());
			else if (key == "calibration_xy")
				command.addParameter(key, _json[tmp][key].get<float>());
			else if (key == "calibration_z")
				command.addParameter(key, _json[tmp][key].get<float>());
			else if (key == "calibration_t")
				command.addParameter(key, _json[tmp][key].get<float>());
			else if (key == "separator")
				command.addParameter(key, _json[tmp][key].get<char>());
			else
				command.addParameter(key, _json[tmp][key].get<size_t>());
		}

		execute(&command);
	}
	else if (tmp == "close") {
		execute(&poca::core::CommandInfo(false, tmp));
	}
	else if (tmp == "computeColocalization") {
		if (_json[tmp].contains("datasetNames")) {
			std::vector <std::string> val = _json[tmp]["datasetNames"].get<std::vector <std::string>>();
			execute(&poca::core::CommandInfo(false, tmp, "datasetNames", val));
		}
	}
	else if (tmp == "changeViewerSize") {
		if (_json[tmp].contains("width") && _json[tmp].contains("height")) {
			uint32_t w = _json[tmp]["width"].get<uint32_t>();
			uint32_t h = _json[tmp]["height"].get<uint32_t>();
			execute(&poca::core::CommandInfo(false, tmp, "width", w, "height", h));
		}
	}
	else if (tmp == "savePositionCamera" || tmp == "loadPositionCamera") {
		poca::core::CommandInfo command(false, tmp);
		if (_json[tmp].contains("path"))
			command.addParameter("path", _json[tmp]["path"].get<std::string>());
		if (_json[tmp].contains("reset"))
			command.addParameter("reset", _json[tmp]["reset"].get<bool>());
		if (_json[tmp].contains("view"))
			command.addParameter("view", _json[tmp]["view"].get<bool>());
		if (_json[tmp].contains("rotation"))
			command.addParameter("rotation", _json[tmp]["rotation"].get<bool>());
		if (_json[tmp].contains("translation"))
			command.addParameter("translation", _json[tmp]["translation"].get<bool>());
		if (_json[tmp].contains("zoom"))
			command.addParameter("zoom", _json[tmp]["zoom"].get<bool>());
		if (_json[tmp].contains("crop"))
			command.addParameter("crop", _json[tmp]["crop"].get<bool>());
		execute(&command);
	}
	else if (tmp == "computeDensityWithRadius") {
		if (_json[tmp].contains("radius")) {
			float val = _json[tmp]["radius"].get<float>();
			execute(&poca::core::CommandInfo(false, tmp, "radius", val));
		}
	}
	else if (tmp == "computeTimingVoronoi") {
		poca::core::MyObjectInterface* object = m_currentMdi->getWidget()->getObject();
		poca::core::BasicComponentInterface* bc = object->getBasicComponent("DetectionSet");
		if (bc == NULL)
			return;
		poca::geometry::DetectionSet* dset = (poca::geometry::DetectionSet*)bc;
		poca::geometry::VoronoiDiagramFactoryInterface* factoryVoronoi = poca::geometry::createVoronoiDiagramFactory();
		clock_t t1, t2;
		t1 = clock();
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::PluginList* plugins = engine->getPlugins();
		for (auto n = 0; n < 50; n++) {
			poca::geometry::VoronoiDiagram* voro = factoryVoronoi->createVoronoiDiagram(object, true, plugins, false);
			if (voro == NULL) return;
			voro->executeCommand(&poca::core::CommandInfo(false, "densityFactor", "factor", 1.6f));
			const std::vector <bool>& selection = voro->getSelection();
			poca::geometry::ObjectIndicesFactoryInterface* factory = poca::geometry::createObjectIndicesFactory();
			std::vector <uint32_t> indices = factory->createObjects(object, selection, (size_t)3);
		}
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("time for computeTimingVoronoi: %ld ms\n", elapsed);
		/*std::vector <float> clusterIndices(indices.size());
		std::transform(indices.begin(), indices.end(), clusterIndices.begin(), [](uint32_t x) { return (float)x; });
		std::map <std::string, poca::core::MyData*>& data = dset->getData();
		data["clustersIndices"] = new poca::core::MyData(clusterIndices);

		object->notifyAll("LoadObjCharacteristicsDetectionSetWidget");*/
		delete factoryVoronoi;

		std::ofstream fs("e:/timings.txt", std::fstream::out | std::fstream::app);
		fs << elapsed << std::endl;
		fs.close();
	}
}