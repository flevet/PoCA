/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MainWindow.cpp
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
#include <QtCore/QMimeData>

#include <OpenGL/Camera.hpp>
#include <Geometry/DetectionSet.hpp>
#include <Interfaces/DelaunayTriangulationInterface.hpp>
#include <Interfaces/DelaunayTriangulationFactoryInterface.hpp>
#include <Interfaces/VoronoiDiagramFactoryInterface.hpp>
#include <Geometry/VoronoiDiagram.hpp>
#include <Plot/Icons.hpp>
#include <DesignPatterns/MediatorWObjectFWidget.hpp>
#include <General/BasicComponent.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/DelaunayTriangulationInterface.hpp>
#include <General/Misc.h>
#include <General/PluginList.hpp>
#include <General/PythonInterpreter.hpp>
#include <DesignPatterns/ListDatasetsSingleton.hpp>
#include <DesignPatterns/StateSoftwareSingleton.hpp>
#include <DesignPatterns/MacroRecorderSingleton.hpp>
#include <Objects/MyObjectDisplayCommand.hpp>
#include <OpenGL/Helper.h>
#include <General/MyData.hpp>
#include <Interfaces/MyObjectInterface.hpp>

#include "../../include/LoaderLocalizationsInterface.hpp"
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
	QSurfaceFormat format;
	format.setVersion(2, 1);
	format.setProfile(QSurfaceFormat::CoreProfile);
	QSurfaceFormat::setDefaultFormat(format);

	loadPlugin();


	std::map <std::string, std::any> singletons;
	poca::core::initializeAllSingletons(singletons);
	poca::opengl::HelperSingleton* help = poca::opengl::HelperSingleton::instance();
	singletons["HelperSingleton"] = help;

	m_plugins->setSingletons(singletons);

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

	poca::core::MediatorWObjectFWidget * mediator = poca::core::MediatorWObjectFWidget::instance();
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
	QObject::connect(m_mfw, SIGNAL(savePosition()), this, SLOT(savePositionCameraSlot()));
	QObject::connect(m_mfw, SIGNAL(loadPosition()), this, SLOT(loadPositionCameraSlot()));

	m_macroW = new MacroWidget(mediator, m_tabWidget);
	mediator->addWidget(m_macroW);
	m_tabWidget->addTab(m_macroW, QObject::tr("Macro"));
	poca::core::MacroRecorderSingleton* macroRecord = std::any_cast <poca::core::MacroRecorderSingleton*>(singletons.at("MacroRecorderSingleton"));
	macroRecord->setTextEdit(m_macroW->getTextEdit());
	macroRecord->setJson(m_macroW->getJson());
	m_macroW->loadParameters(std::any_cast <poca::core::StateSoftwareSingleton*>(singletons.at("StateSoftwareSingleton"))->getParameters());

	for (size_t n = 0; n < m_GUIWidgets.size(); n++)
		m_GUIWidgets[n]->addGUI(mediator, m_tabWidget);
	m_plugins->addGUI(mediator, m_tabWidget);

#ifndef NO_PYTHON
	m_pythonW = new PythonWidget(mediator, m_tabWidget);
	mediator->addWidget(m_pythonW);
	m_tabWidget->addTab(m_pythonW, QObject::tr("Python"));
	m_pythonW->loadParameters(std::any_cast <poca::core::StateSoftwareSingleton*>(singletons.at("StateSoftwareSingleton"))->getParameters());
#endif

	m_ROIsW = new ROIGeneralWidget(mediator, m_tabWidget);
	mediator->addWidget(m_ROIsW);
	poca::core::utils::addWidget(m_tabWidget, QString("ROI Manager"), QString("General"), m_ROIsW, false);

	//m_tabWidget->addTab(m_ROIsW, QObject::tr("ROI Manager"));

	for (int n = 0; n < m_tabWidget->count(); n++) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
		m_tabWidget->setTabVisible(n, m_tabWidget->tabText(n) == "Misc." || m_tabWidget->tabText(n) == "Macro");
#endif
		QTabWidget* tab = dynamic_cast <QTabWidget*>(m_tabWidget->widget(n));
		if (!tab) continue;
		tab->setCurrentIndex(0);
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

	setWindowTitle(tr("PoCA: Point Cloud Analyst - v0.5.0"));
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
	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	nlohmann::json& parameters = sss->getParameters();

	if (m_currentMdi != NULL) {
		m_currentMdi->getWidget()->getObject()->saveCommands(parameters);
	}
	poca::core::CommandInfo command(false, "saveParameters", "file", &parameters);
	m_plugins->execute(&command);
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
	m_openDirAct->setStatusTip(tr("Open aqn existing directory"));
	QObject::connect(m_openDirAct, SIGNAL(triggered()), this, SLOT(openDir()));

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

	QActionGroup* roiGroup = new QActionGroup(this);
	roiGroup->setExclusionPolicy(QActionGroup::ExclusionPolicy::ExclusiveOptional);
	roiGroup->addAction(m_line2DROIAct);
	roiGroup->addAction(m_triangle2DROIAct);
	roiGroup->addAction(m_circle2DROIAct);
	roiGroup->addAction(m_square2DROIAct);
	roiGroup->addAction(m_polyline2DROIAct);
	roiGroup->addAction(m_sphere3DROIAct);
}

void MainWindow::createMenus()
{
	QMenuBar* menuB = menuBar();

	QMenu* fileMenu = menuB->addMenu("File");
	QMenu* openMenu = fileMenu->addMenu("Open");
	openMenu->addAction(m_openFileAct);
	openMenu->addAction(m_openDirAct);

	const std::vector <PluginInterface*>& plugins = m_plugins->getPlugins();
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
	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	nlohmann::json& parameters = sss->getParameters();

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
	m_fileToolBar->addAction(m_duplicateAct);
	m_fileToolBar->addSeparator();
	m_lastActionQuantifToolbar = m_fileToolBar->addSeparator();
	m_fileToolBar->addAction(m_line2DROIAct);
	m_fileToolBar->addAction(m_triangle2DROIAct);
	m_fileToolBar->addAction(m_circle2DROIAct);
	m_fileToolBar->addAction(m_square2DROIAct);
	m_fileToolBar->addAction(m_polyline2DROIAct);
	m_fileToolBar->addAction(m_sphere3DROIAct);
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

	const std::vector <PluginInterface*>& plugins = m_plugins->getPlugins();
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
	QObject* sender = QObject::sender();
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
}

void MainWindow::setCameraInteraction(bool _on)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
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
	else if (sender == m_cropAct)
		cam->setCameraInteraction(_on ? poca::opengl::Camera::Crop : poca::opengl::Camera::None);

	if (act != NULL)
		m_cropAct->setEnabled(act->isChecked());
	if(!m_xyAct->isChecked() && !m_xzAct->isChecked() && !m_yzAct->isChecked() && cam->getCameraInteraction() == poca::opengl::Camera::Crop)
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
	obj = m_plugins->actionTriggered(sender, obj);

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
		//if (!name.endsWith(".txt"))
		execute(&poca::core::CommandInfo(true, "open", "path", name.toStdString()));
		if (name.endsWith(".txt")) {
			poca::core::CommandInfo ci(true, "openFile", "name", name.toStdString());
			m_plugins->execute(&ci);
			if (!ci.hasParameter("object")) continue;
			poca::core::MyObjectInterface* obj = ci.getParameterPtr<poca::core::MyObjectInterface>("object");
			if (obj != NULL) {
				createWidget(obj);
			}
		}
	}
}

void MainWindow::createWidget(poca::core::MyObjectInterface* _obj)
{
	poca::opengl::Camera* cam = new poca::opengl::Camera(_obj, _obj->dimension(), NULL);// this);

	poca::core::MediatorWObjectFWidget* mediator = poca::core::MediatorWObjectFWidget::instance();
	poca::core::SubjectInterface* subject = dynamic_cast<poca::core::SubjectInterface*>(_obj);
	if (subject) {
		mediator->addObserversToSubject(subject, "LoadObjCharacteristicsAllWidgets");
		mediator->addObserversToSubject(subject, "LoadObjCharacteristicsMiscWidget");
		mediator->addObserversToSubject(subject, "LoadObjCharacteristicsDetectionSetWidget");
		mediator->addObserversToSubject(subject, "LoadObjCharacteristicsDelaunayTriangulationWidget");
		mediator->addObserversToSubject(subject, "LoadObjCharacteristicsVoronoiDiagramWidget");
	}
	_obj->attach(cam, "updateDisplay");
	_obj->attach(cam, "updateInfosObject");
	_obj->attach(this, "addCommandLastAddedComponent");
	_obj->attach(this, "addCommandToSpecificComponent");
	_obj->attach(this, "duplicateCleanedData"); 

	_obj->addCommand(new MyObjectDisplayCommand(_obj));
	poca::core::BasicComponent* bci = _obj->getLastAddedBasicComponent();
	if (bci != NULL && bci->nbCommands() == 0)
		m_plugins->addCommands(bci);

	MdiChild* child = new MdiChild(cam);
	QObject::connect(child, SIGNAL(setCurrentMdi(MdiChild*)), this, SLOT(setActiveMdiChild(MdiChild*)));
	m_mdiArea->addSubWindow(child);
	setActiveMdiChild(child);
	child->layout()->update();
	child->layout()->activate();
	child->getWidget()->update();
	child->show();

	_obj->notify("LoadObjCharacteristicsAllWidgets");
	_obj->notifyAll("updateDisplay");
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

void MainWindow::openFile(const QString& _filename, poca::core::CommandInfo* _command)
{
	std::map <std::string, std::vector <float>> data;
	for (std::size_t n = 0; n < m_loaderslocsFile.size() && data.empty(); n++)
		m_loaderslocsFile.at(n)->loadFile(_filename, data, _command);

	if (data.empty()) return;

	poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(data);

	QString dir, name;
	decomposePathToDirAndFile(_filename, dir, name);

	createWindows(dset, dir, name);
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
	poca::core::BasicComponent* bci = obj->getBasicComponent("DetectionSet");
	poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
	if (dset == NULL) return;
	poca::geometry::DetectionSet* newDset = dset->duplicateSelection();
	const std::string& dir = obj->getDir(), name = obj->getName();
	QString newName(name.c_str());
	int index = newName.lastIndexOf(".");
	newName.insert(index, QString("_%1").arg(m_currentDuplicate++));
	createWindows(newDset, QString(dir.c_str()), newName);
}

void MainWindow::createWindows(poca::geometry::DetectionSet* _dset, const QString& _dir, const QString& _name)
{
	if (_dset == NULL) return;

	poca::core::MyObject* wobj = new SMLMObject();
	wobj->setDir(_dir.toLatin1().data());
	wobj->setName(_name.toLatin1().data());
	wobj->addBasicComponent(_dset);
	wobj->setDimension(_dset->dimension());

	if (wobj != NULL) {
		poca::opengl::Camera* cam = new poca::opengl::Camera(wobj, _dset->dimension(), NULL);// this);

		int indexVoronoiTab = 0;

		poca::core::MediatorWObjectFWidget* mediator = poca::core::MediatorWObjectFWidget::instance();

		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsAllWidgets");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsMiscWidget");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsDetectionSetWidget");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsDelaunayTriangulationWidget");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsVoronoiDiagramWidget");
		wobj->attach(cam, "updateDisplay");
		wobj->attach(cam, "updateInfosObject");

		wobj->attach(this, "addCommandLastAddedComponent");
		wobj->attach(this, "addCommandToSpecificComponent");
		wobj->attach(this, "duplicateCleanedData");
		
		SMLMObject* sobj = dynamic_cast <SMLMObject*>(wobj);
		sobj->addCommand(new MyObjectDisplayCommand(sobj));

		m_plugins->addCommands(_dset);
		m_plugins->addCommands(wobj);

		MdiChild* child = new MdiChild(cam);
		QObject::connect(child, SIGNAL(setCurrentMdi(MdiChild*)), this, SLOT(setActiveMdiChild(MdiChild*)));
		m_mdiArea->addSubWindow(child);
		setActiveMdiChild(child);

		child->layout()->update();
		child->layout()->activate();
		child->getWidget()->update();
		child->show();

		poca::geometry::DelaunayTriangulationFactoryInterface* factory = poca::geometry::createDelaunayTriangulationFactory();
		poca::geometry::DelaunayTriangulationInterface* delaunay = factory->createDelaunayTriangulation(wobj, NULL, false);
		delete factory;
	}
}

void MainWindow::setActiveMdiChild(MdiChild * _mdiChild)
{
	poca::core::MediatorWObjectFWidget* mediator = poca::core::MediatorWObjectFWidget::instance();
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

		//m_xyAct->setEnabled(dimension == 3);
		m_xzAct->setEnabled(dimension == 3);
		m_yzAct->setEnabled(dimension == 3);

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

void MainWindow::loadPlugin()
{
	m_plugins = new poca::core::PluginList();
	QDir pluginsDir(QCoreApplication::applicationDirPath());
#if defined(Q_OS_WIN)
	if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release")
		pluginsDir.cdUp();
#elif defined(Q_OS_MAC)
	if (pluginsDir.dirName() == "MacOS") {
		pluginsDir.cdUp();
		pluginsDir.cdUp();
		pluginsDir.cdUp();
	}
#endif
	pluginsDir.cd("plugins");
	QString extension(".dll");
#if defined _DEBUG
	pluginsDir.cd("Debug");
	extension.push_front("d");
#endif
	//std::cout << pluginsDir.absolutePath().toLatin1().data() << std::endl;
	const QStringList entries = pluginsDir.entryList(QDir::Files);
	for (const QString& fileName : entries) {
		if (!fileName.endsWith(extension)) continue;
		QPluginLoader pluginLoader(pluginsDir.absoluteFilePath(fileName));
		QObject* plugin = pluginLoader.instance();
		LoaderLocalizationsInterface* linterface = NULL;
		GuiInterface* ginterface = NULL;
		PluginInterface* pinterface = NULL;
		if (plugin) {
			linterface = qobject_cast<LoaderLocalizationsInterface*>(plugin);
			if (linterface)
				m_loaderslocsFile.push_back(linterface);
			ginterface = qobject_cast<GuiInterface*>(plugin);
			if (ginterface)
				m_GUIWidgets.push_back(ginterface);
			pinterface = qobject_cast<PluginInterface*>(plugin);
			if (pinterface)
				m_plugins->addPlugin(pinterface);
			if(linterface == NULL && ginterface == NULL && pinterface == NULL)
				pluginLoader.unload();
		}
	}
	const std::vector <PluginInterface*>& plugs = m_plugins->getPlugins();
	for (PluginInterface* plugin : plugs)
		plugin->setPlugins(m_plugins);
}

void MainWindow::update(poca::core::SubjectInterface* _subj, const poca::core::CommandInfo& _action)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*>(_subj);
	if (obj == NULL) return;

	if(_action == "addCommandToSpecificComponent"){
		poca::core::BasicComponent* comp = _action.getParameterPtr<poca::core::BasicComponent>("component");
		m_plugins->addCommands(comp);
		poca::core::ListDatasetsSingleton::instance()->getObject(obj)->notify("LoadObjCharacteristicsAllWidgets");
		updateTabWidget();
		obj->notify("LoadObjCharacteristicsAllWidgets");
	}
	if (_action == "addCommandLastAddedComponent") {
		poca::core::BasicComponent* bci = obj->getLastAddedBasicComponent();
		if (bci == NULL) return;
		m_plugins->addCommands(bci);

		poca::core::ListDatasetsSingleton::instance()->getObject(obj)->notify("LoadObjCharacteristicsAllWidgets");

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
	cam->repaint();
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

	for (poca::core::MyObjectInterface* obj : objs)
		if (obj == NULL) return;

	poca::core::ChangeManagerSingleton* singleton = poca::core::ChangeManagerSingleton::instance();

	ColocObject* wobj = new ColocObject(objs);
	wobj->setDir(objs[0]->getDir());
	QString name("Colocalization_[");
	for (poca::core::MyObjectInterface* obj : objs)
		name.append(obj->getName().c_str()).append(",");
	name.append("]");
	wobj->setName(name.toLatin1().data());
	if (wobj != NULL) {
		std::vector <std::string> names;
		for (MdiChild* mdi : _ws)
			names.push_back(mdi->getWidget()->getObject()->getName());
		poca::core::MacroRecorderSingleton::instance()->addCommand("MainWindow", &poca::core::CommandInfo(true, "computeColocalization", "datasetNames", names));

		poca::core::CommandInfo ciHeatmap(false, "DetectionSet", "displayHeatmap", false);
		poca::core::CommandInfo ci(false, "All", "freeGPU", true);
		for (size_t n = 0; n < objs.size(); n++) {
			objs[n]->executeCommandOnSpecificComponent("DetectionSet", &poca::core::CommandInfo(false, "displayHeatmap", false));
			objs[n]->executeGlobalCommand(&poca::core::CommandInfo(false, "freeGPU"));
			poca::core::SubjectInterface* subject = dynamic_cast <poca::core::SubjectInterface*>(objs[n]);
			if(subject)
				singleton->UnregisterFromAllObservers(subject);
		}
		for (MdiChild* mc : _ws) {
			poca::opengl::CameraInterface* camW = mc->getWidget();
			camW->setDeleteObject(false);
			camW->makeCurrent();
			m_mdiArea->removeSubWindow(mc);
			delete mc;
		}

		poca::opengl::Camera* cam = new poca::opengl::Camera(wobj, wobj->dimension(), this);

		int indexVoronoiTab = 0;

		poca::core::MediatorWObjectFWidget* mediator = poca::core::MediatorWObjectFWidget::instance();

		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsAllWidgets");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsMiscWidget");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsDetectionSetWidget");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsDelaunayTriangulationWidget");
		mediator->addObserversToSubject(wobj, "LoadObjCharacteristicsVoronoiDiagramWidget");
		wobj->attach(cam, "updateDisplay");
		wobj->attach(cam, "updateInfosObject");
		wobj->attach(cam, "updateInfosObjectOverlap");

		wobj->attach(this, "addCommandLastAddedComponent");

		wobj->addCommand(new MyObjectDisplayCommand(wobj));
		m_plugins->addCommands(wobj);

		for (size_t n = 0; n < objs.size(); n++) {
			objs[n]->attach(cam, "updateDisplay");
			objs[n]->attach(this, "addCommandLastAddedComponent");
			objs[n]->attach(this, "addCommandToSpecificComponent");
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

void MainWindow::savePositionCameraSlot()
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
	json["translationMatrix"] = cam->getTranslationMatrix();
	json["distanceOrtho"] = cam->getDistanceOrtho();
	json["distanceOrthoOriginal"] = cam->getOriginalDistanceOrtho();
	json["crop"] = cam->getCurrentCrop();

	std::string text = json.dump();
	std::cout << text << std::endl;
	std::ofstream fs(filename.toLatin1().data());
	fs << text;
	fs.close();
}

void MainWindow::loadPositionCameraSlot()
{
	execute(&poca::core::CommandInfo(false, "loadPositionCamera"));
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
	execute(&poca::core::CommandInfo(true, "loadPositionCamera", "path", filename.toStdString()));
}

void MainWindow::loadPositionCamera(const std::string& _filename, const bool _reset)
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
		if (tmp.contains("matrixView"))
			stateCam.m_matrixView = tmp["matrixView"].get<glm::mat4>();
		if (tmp.contains("rotationSum"))
			stateCam.m_rotationSum = tmp["rotationSum"].get<glm::quat>();
		if (tmp.contains("rotation"))
			stateCam.m_rotation = tmp["rotation"].get<glm::quat>();
		if (tmp.contains("center"))
			stateCam.m_center = tmp["center"].get<glm::vec3>();
		if (tmp.contains("eye"))
			stateCam.m_eye = tmp["eye"].get<glm::vec3>();
		if (tmp.contains("up"))
			stateCam.m_up = tmp["up"].get<glm::vec3>();
	}
	if (json.contains("translationMatrix"))
		cam->setTranslationMatrix(json["translationMatrix"].get<glm::mat4>());
	if (json.contains("distanceOrtho"))
		cam->setDistanceOrtho(json["distanceOrtho"].get<float>());
	if (json.contains("crop"))
		cam->setCurrentCrop(json["crop"].get<poca::core::BoundingBox>());

	cam->zoomToBoundingBox(cam->getCurrentCrop(), _reset);
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
		if (_com->hasParameter("path")) {
			std::string filename = _com->getParameter<std::string>("path");
			std::cout << "reset " << reset << std::endl;
			loadPositionCamera(filename, reset);
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
					if (command["open"].contains("path"))
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
						if (!command.empty())
							comObj->executeCommand(&command);
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
		execute(&command);
	}
	else if (tmp == "computeDensityWithRadius") {
		if (_json[tmp].contains("radius")) {
			float val = _json[tmp]["radius"].get<float>();
			execute(&poca::core::CommandInfo(false, tmp, "radius", val));
		}
	}
}