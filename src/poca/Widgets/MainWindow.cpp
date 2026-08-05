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

//#define STB_RECT_PACK_IMPLEMENTATION

#include <Windows.h>
#include <gl/glew.h>
#include <iostream>
#include <limits>
#include <random>
#include <cmath>
#include <fstream>
#include <functional>
#include <QtCore/QSignalMapper>
#include <QtCore/QDir>
#include <QtCore/QPluginLoader>
#include <QtCore/QVariant>
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
#include <QtWidgets/QAbstractItemView>
#include <QtWidgets/QFrame>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTableWidgetItem>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItem>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QTabBar>
#include <QtGui/QDragEnterEvent>
#include <QtGui/QDropEvent>
#include <QtGui/QImage>
#include <QtGui/QBrush>
#include <QtGui/QColor>
#include <QtGui/QFont>
#include <QtCore/QMimeData>
#include <qmath.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Polygon_mesh_processing/angle_and_area_smoothing.h>
#include <CGAL/boost/graph/copy_face_graph.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Aff_transformation_2.h>

#include <OpenGL/Camera.hpp>
#include <Geometry/DetectionSet.hpp>
#include <Geometry/GeometryCommandContext.hpp>
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
#include <Objects/ObjectCommandContext.hpp>
#include <General/JsonCommandContext.hpp>
#include <OpenGL/Helper.h>
#include <General/MyData.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/ObjectIndicesFactoryInterface.hpp>
#include <General/Image.hpp>
#include <General/Engine.hpp>
#include <General/Palette.hpp>
#include <Geometry/ObjectLists.hpp>
#include <Geometry/ObjectListMesh.hpp>
#include <General/ImagesList.hpp>
#include <Cuda/BasicOperationsImage.h>
#include <Cuda/ConnectedComponents.h>
#include <Geometry/CGAL_helpers.hpp>
#include <General/stb_rect_pack.h>
#include <Objects/MyMultipleObject.hpp>
#include <Geometry/ObjectListPolygon.hpp>
#include <General/Misc.h>
#include "../../include/GuiInterface.hpp"
#include "../../include/PluginInterface.hpp"

#include "../Widgets/MainWindow.hpp"
#include "../Widgets/MdiChild.hpp"
#include "../Objects/SMLM_Object/SMLMObject.hpp"
#include "../Objects/Coloc_Object/ColocObject.hpp"
#include "../Widgets/MainFilterWidget.hpp"
#include "../Widgets/ColocalizationChoiceDialog.hpp"
#include "../Widgets/MergeDatasetsChoiceDialog.hpp"
#include "../Widgets/PythonWidget.hpp"
#include "../Widgets/ROIGeneralWidget.hpp"
#include "../Widgets/MacroWidget.hpp"
#include "../Widgets/ReorganizeRenderingWidget.hpp"
#include <Widgets/PerformanceWidget.hpp>
#include "../Widgets/ColorButtonGridWidget.hpp"
#include "../Widgets/DatasetAssemblerWidget.hpp"
#include <Widgets/CustomColorDialog.hpp>
#include "../../poca_voronoidiagramplugin/VoronoiCommandContext.hpp"

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
	m_mdiArea->setViewMode(QMdiArea::SubWindowView);

	setCentralWidget(m_mdiArea);
	m_windowMapper = new QSignalMapper(this);

	createActions();
	createToolBars();
	createMenus();
	createStatusBar();

	m_tabWidget = new QTabWidget(this);
	m_tabWidget->setObjectName("TabWidget");
	m_tabWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_tabWidget->setContentsMargins(0, 0, 0, 0);
	configureInspectorTabWidget(m_tabWidget);
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

	ReorganizeRenderingWidget* rrw = new ReorganizeRenderingWidget(mediator, m_tabWidget);
	mediator->addWidget(rrw);
	poca::core::utils::addWidget(m_tabWidget, QString("Misc."), QString("Reorganize rendering"), rrw, false);

	m_macroW = new MacroWidget(mediator, this);
	mediator->addWidget(m_macroW);
	macroRecord->setTextEdit(m_macroW->getTextEdit());
	macroRecord->setJson(m_macroW->getJson());
	m_macroW->loadParameters(engine->getGlobalParameters());

	m_datasetAssemblerW = new DatasetAssemblerWidget(this);
	m_datasetAssemblerW->loadParameters(engine->getGlobalParameters());
	QObject::connect(m_datasetAssemblerW, SIGNAL(transferNewObjectCreated(poca::core::MyObjectInterface*)), this, SLOT(createWidget(poca::core::MyObjectInterface*)));

	m_ROIsW = new ROIGeneralWidget(mediator, this);
	mediator->addWidget(m_ROIsW);

	engine->addGUI(m_tabWidget);

#ifndef NO_PYTHON
	m_pythonW = new PythonWidget(mediator, this);
	m_pythonW->setWindowFlags(m_pythonW->windowFlags() | Qt::Window);
	m_pythonW->setWindowTitle(tr("Python"));
	m_pythonW->resize(420, 320);
	mediator->addWidget(m_pythonW);
	m_pythonW->loadParameters(engine->getGlobalParameters());
	m_pythonW->hide();
#endif

	for (int n = 0; n < m_tabWidget->count(); n++) {
/*#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
		m_tabWidget->setTabVisible(n, m_tabWidget->tabText(n) == "Misc." || m_tabWidget->tabText(n) == "Macro");
#endif*/
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

	if (m_tabWidget->count() > 0)
		m_tabWidget->setCurrentIndex(0);

	m_widgetColors = new ColorButtonGridWidget(this);
	m_widgetColors->setMaxPerRow(20);
	m_widgetColors->setRightButtonText("+");
	m_widgetColors->setVisible(false);

	QObject::connect(m_widgetColors, SIGNAL(indexClicked(int)), this, SLOT(changeColorObject(int)));
	QObject::connect(m_widgetColors, &ColorButtonGridWidget::rightButtonClicked, this, 
		[]() {
			poca::core::Engine::instance()->toggleGlobalCommands();
		}
	);

	//connect(grid, &ColorButtonGridWidget::rightButtonClicked, this, &YourClass::onAddObject);

	/*QHBoxLayout* layoutColor = new QHBoxLayout;
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
	QObject::connect(m_colorButtonsGroup, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(changeColorObject(QAbstractButton*)));*/

	createDesignDock();

	setActiveMdiChild(NULL);

	setWindowTitle(tr("PoCA: Point Cloud Analyst"));
	setUnifiedTitleAndToolBarOnMac(true);
	statusBar()->showMessage(tr("Ready"));
	m_lblPermanentStatus = new QLabel;
	m_lblPermanentStatus->setVisible(false);
	m_progressBar = new QProgressBar(this);
	m_progressBar->setVisible(false);
	m_statusObjectLabel = new QLabel(tr("Objects: 0"), this);
	m_statusBoxLabel = new QLabel(tr("Box: -"), this);
	statusBar()->addPermanentWidget(m_lblPermanentStatus);
	statusBar()->addPermanentWidget(m_statusObjectLabel);
	statusBar()->addPermanentWidget(m_statusBoxLabel);
	statusBar()->addPermanentWidget(m_progressBar);
	applyPrototypeStyle();
	refreshObjectsPanel();
	refreshPropertiesPanel();

	poca::core::NbObjects = m_mfw->getFirstIndexObj();

	setAcceptDrops(true);

	QObject::connect(m_macroW, SIGNAL(runMacro(std::vector<nlohmann::json>, bool)), this, SLOT(runMacro(std::vector<nlohmann::json>, bool)));
	QObject::connect(m_macroW, SIGNAL(runMacro(std::vector<nlohmann::json>, QStringList)), this, SLOT(runMacro(std::vector<nlohmann::json>, QStringList)));
}

MainWindow::~MainWindow()
{
	nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();

	if (m_currentMdi != NULL) {
		m_currentMdi->getWidget()->getObject()->saveCommands(parameters);
	}
	poca::core::CommandInfo command(false, "saveParameters");
	poca::core::CommandExecutionContext runtimeContext;
	runtimeContext.set(poca::core::JsonFileContext{ &parameters });
	poca::core::Engine* engine = poca::core::Engine::instance();
	engine->getPlugins()->execute(&command, runtimeContext);
	m_macroW->execute(&command, runtimeContext);
	m_datasetAssemblerW->saveParameters(parameters);
	if (m_pythonW != NULL)
		m_pythonW->execute(&command, runtimeContext);
	parameters["Preferences"]["verbose"] = engine->verboseEnabled();
	parameters["Preferences"]["verboseTypes"] = engine->verboseTypes();
	engine->savePalettesToGlobalParameters();

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
	poca::core::Engine* engine = poca::core::Engine::instance();

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

	m_datasetAssemblerAct = new QAction(QIcon(QPixmap(poca::plot::openDirIcon)), tr("Dataset assembler"), this);
	m_datasetAssemblerAct->setStatusTip(tr("Open the dataset assembler"));
	QObject::connect(m_datasetAssemblerAct, SIGNAL(triggered()), this, SLOT(openDatasetAssembler()));

#ifndef NO_PYTHON
	m_pythonWidgetAct = new QAction(QIcon(QPixmap(poca::plot::openFileIcon)), tr("Python"), this);
	m_pythonWidgetAct->setStatusTip(tr("Open the Python widget"));
	QObject::connect(m_pythonWidgetAct, SIGNAL(triggered()), this, SLOT(openPythonWidget()));
#endif

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

	m_reloadDatasetAct = new QAction(QIcon(QPixmap(poca::plot::invertIcon)), tr("Reload dataset"), this);//
	m_reloadDatasetAct->setStatusTip(tr("Reload dataset"));
	QObject::connect(m_reloadDatasetAct, SIGNAL(triggered()), this, SLOT(reloadCurrentDataset()));

	m_aboutAct = new QAction(QIcon("./images/about.png"), tr("About..."), this);
	m_aboutAct->setStatusTip(tr("About..."));
	QObject::connect(m_aboutAct, SIGNAL(triggered()), this, SLOT(aboutDialog()));

	m_exitAct = new QAction(tr("E&xit"), this);
	m_exitAct->setShortcuts(QKeySequence::Quit);
	m_exitAct->setStatusTip(tr("Exit the application"));
	connect(m_exitAct, SIGNAL(triggered()), qApp, SLOT(closeAllWindows()));

	m_verboseAct = new QAction(tr("Verbose"), this);
	m_verboseAct->setCheckable(true);
	m_verboseAct->setChecked(engine->verboseEnabled());
	m_verboseAct->setStatusTip(tr("Enable verbose output"));
	QObject::connect(m_verboseAct, SIGNAL(toggled(bool)), this, SLOT(toggleVerbose(bool)));

	m_addVerboseTypeAct = new QAction(tr("Add verbose type..."), this);
	m_addVerboseTypeAct->setStatusTip(tr("Add a verbose output type filter"));
	QObject::connect(m_addVerboseTypeAct, SIGNAL(triggered()), this, SLOT(addVerboseType()));

	m_clearVerboseTypesAct = new QAction(tr("Clear verbose types"), this);
	m_clearVerboseTypesAct->setStatusTip(tr("Clear verbose output type filters"));
	QObject::connect(m_clearVerboseTypesAct, SIGNAL(triggered()), this, SLOT(clearVerboseTypes()));

	m_palettesAct = new QAction(tr("Palettes"), this);
	m_palettesAct->setStatusTip(tr("Edit application palettes"));
	QObject::connect(m_palettesAct, SIGNAL(triggered()), this, SLOT(openPalettesDialog()));

	m_performanceWidgetAct = new QAction(tr("Performance"), this);
	m_performanceWidgetAct->setCheckable(true);
	m_performanceWidgetAct->setChecked(false);
	m_performanceWidgetAct->setStatusTip(tr("Show performance monitor"));
	QObject::connect(m_performanceWidgetAct, SIGNAL(toggled(bool)), this, SLOT(togglePerformanceWidget(bool)));

	m_debugPyramidalRenderingAct = new QAction(tr("debugPyramidalRendering"), this);
	m_debugPyramidalRenderingAct->setCheckable(true);
	m_debugPyramidalRenderingAct->setChecked(engine->hasVerboseType("debugPyramidalRendering"));
	m_debugPyramidalRenderingAct->setStatusTip(tr("Print pyramidal rendering diagnostics"));
	QObject::connect(m_debugPyramidalRenderingAct, SIGNAL(toggled(bool)), this, SLOT(toggleDebugPyramidalRendering(bool)));

	m_debugGizmoAct = new QAction(tr("debugGizmo"), this);
	m_debugGizmoAct->setCheckable(true);
	m_debugGizmoAct->setChecked(engine->hasVerboseType("debugGizmo"));
	m_debugGizmoAct->setStatusTip(tr("Print transform gizmo diagnostics"));
	QObject::connect(m_debugGizmoAct, SIGNAL(toggled(bool)), this, SLOT(toggleDebugGizmo(bool)));

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

	m_freehandROIAct = new QAction(QIcon(QPixmap(poca::plot::freehandIcon)), tr("&Freehand"), this);
	m_freehandROIAct->setStatusTip(tr("Freehand ROI"));
	m_freehandROIAct->setCheckable(true);
	connect(m_freehandROIAct, SIGNAL(triggered()), this, SLOT(setCameraInteraction()));

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
	roiGroup->addAction(m_freehandROIAct);
}

void MainWindow::createMenus()
{
	QMenuBar* menuB = menuBar();

	QMenu* fileMenu = menuB->addMenu("File");
	QMenu* openMenu = fileMenu->addMenu("Open");
	openMenu->addAction(m_openFileAct);
	openMenu->addAction(m_openDirAct);
	openMenu->addAction(m_plusAct);

	QMenu* preferencesMenu = menuB->addMenu("Preferences");
	QMenu* verboseMenu = preferencesMenu->addMenu("Verbose");
	verboseMenu->addAction(m_verboseAct);
	verboseMenu->addSeparator();
	verboseMenu->addAction(m_debugPyramidalRenderingAct);
	verboseMenu->addAction(m_debugGizmoAct);
	verboseMenu->addAction(m_addVerboseTypeAct);
	verboseMenu->addAction(m_clearVerboseTypesAct);
	preferencesMenu->addAction(m_performanceWidgetAct);
	preferencesMenu->addAction(m_palettesAct);
	registerTests();
	QMenu* testsMenu = preferencesMenu->addMenu("Tests");
	createTestsMenu(testsMenu);

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

}

void MainWindow::openPalettesDialog()
{
	CustomColorDialog dlg(this);
	dlg.exec();
}

void MainWindow::togglePerformanceWidget(bool _enabled)
{
	if (_enabled) {
		if (m_performanceW == nullptr) {
			m_performanceW = new poca::qt::PerformanceWidget(this);
			m_performanceW->setWindowFlags(m_performanceW->windowFlags() | Qt::Window);
			m_performanceW->setAttribute(Qt::WA_DeleteOnClose, true);
			m_performanceW->setWindowTitle(tr("Performance"));
			m_performanceW->resize(520, 300);
			QObject::connect(m_performanceW, &QWidget::destroyed, this, [this]() {
				m_performanceW = nullptr;
				if (m_performanceWidgetAct != nullptr) {
					m_performanceWidgetAct->blockSignals(true);
					m_performanceWidgetAct->setChecked(false);
					m_performanceWidgetAct->blockSignals(false);
				}
			});
		}
		m_performanceW->show();
		m_performanceW->raise();
		m_performanceW->activateWindow();
	}
	else if (m_performanceW != nullptr) {
		m_performanceW->close();
	}
}

void MainWindow::toggleVerbose(bool _enabled)
{
	poca::core::Engine::instance()->setVerbose(_enabled);
}

void MainWindow::addVerboseType()
{
	bool ok = false;
	const QString type = QInputDialog::getText(this, tr("Verbose type"), tr("Type:"), QLineEdit::Normal, QString(), &ok);
	if (!ok)
		return;

	const QString trimmed = type.trimmed();
	if (trimmed.isEmpty())
		return;

	poca::core::Engine* engine = poca::core::Engine::instance();
	engine->addVerboseType(trimmed.toStdString());
	if (trimmed == "debugPyramidalRendering")
		m_debugPyramidalRenderingAct->setChecked(true);
	else if (trimmed == "debugGizmo")
		m_debugGizmoAct->setChecked(true);
}

void MainWindow::clearVerboseTypes()
{
	poca::core::Engine::instance()->clearVerboseTypes();
	m_debugPyramidalRenderingAct->setChecked(false);
	m_debugGizmoAct->setChecked(false);
}

void MainWindow::toggleDebugPyramidalRendering(bool _enabled)
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	if (_enabled)
		engine->addVerboseType("debugPyramidalRendering");
	else
		engine->removeVerboseType("debugPyramidalRendering");
}

void MainWindow::toggleDebugGizmo(bool _enabled)
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	if (_enabled)
		engine->addVerboseType("debugGizmo");
	else
		engine->removeVerboseType("debugGizmo");
}

void MainWindow::registerTests()
{
	poca::core::Engine* engine = poca::core::Engine::instance();
#ifndef NO_CUDA
	registerFaceConnectedComponentTests(engine->tests());
#endif
	const std::vector <PluginInterface*>& plugins = engine->getPlugins()->getPlugins();
	for (PluginInterface* plugin : plugins)
		plugin->registerTests(engine->tests());
}

void MainWindow::createTestsMenu(QMenu* _testsMenu)
{
	std::map<std::string, QMenu*> menus;
	poca::core::Engine* engine = poca::core::Engine::instance();
	for (const poca::core::TestActionDescriptor& test : engine->tests().descriptors()) {
		QMenu* menu = createTestMenu(_testsMenu, menus, test.menuPath);
		QAction* action = new QAction(QString::fromStdString(test.label), this);
		action->setStatusTip(QString::fromStdString(test.statusTip));
		QObject::connect(action, &QAction::triggered, this, [this, test]() {
			createTestObject(test);
		});
		menu->addAction(action);
	}
}

QMenu* MainWindow::createTestMenu(QMenu* _testsMenu, std::map<std::string, QMenu*>& _menus, const std::string& _menuPath)
{
	if (_menuPath.empty())
		return _testsMenu;
	auto it = _menus.find(_menuPath);
	if (it != _menus.end())
		return it->second;
	QMenu* menu = _testsMenu->addMenu(QString::fromStdString(_menuPath));
	_menus[_menuPath] = menu;
	return menu;
}

void MainWindow::createTestObject(const poca::core::TestActionDescriptor& _test)
{
#ifdef NO_CUDA
	if (_test.requiresCuda) {
		QMessageBox::warning(this, tr("CUDA unavailable"), tr("This test requires CUDA."));
		return;
	}
#endif
	poca::core::MyObjectInterface* obj = _test.createObject();
	if (obj == NULL)
		return;
	poca::opengl::CameraInterface* cam = createWindows(obj);
	poca::core::Engine::instance()->addCameraToObject(obj, cam);
	if (cam != NULL)
		cam->makeCurrent();
	poca::core::CommandInfo ci(false, "createDisplay");
	obj->executeGlobalCommand(&ci);
	obj->notify("LoadObjCharacteristicsAllWidgets");
	obj->notifyAll("updateDisplay");
}

void MainWindow::createToolBars()
{
	m_fileToolBar = new QToolBar(tr("Toolbar"));
	m_fileToolBar->addAction(m_openFileAct);
	m_fileToolBar->addAction(m_openDirAct);
	m_fileToolBar->addAction(m_plusAct);
	m_fileToolBar->addAction(m_datasetAssemblerAct);
#ifndef NO_PYTHON
	m_fileToolBar->addAction(m_pythonWidgetAct);
#endif
	m_fileToolBar->addAction(m_duplicateAct);
	m_fileToolBar->addAction(m_reloadDatasetAct);
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
	m_fileToolBar->addAction(m_freehandROIAct);
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

	addToolBar(Qt::TopToolBarArea, m_fileToolBar);
}

void MainWindow::setCameraInteraction()
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	size_t dimension = cam->getObject()->dimension();
	QObject* sender = QObject::sender();
	if (sender == m_xyAct) {
		return;
		/*if (dimension == 3)
			return;
		else {
			cam->fixPlane(poca::opengl::Camera::Plane_XY, true);
			cam->fixPlane(poca::opengl::Camera::Plane_XY, false);
			return;
		}*/
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
	else if (sender == m_freehandROIAct)
		cam->setCameraInteraction(m_freehandROIAct->isChecked() ? poca::opengl::Camera::FreehandDefinition : poca::opengl::Camera::None);
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
		m_line2DROIAct->setEnabled(true);// !_on || dimension == 2);
		m_triangle2DROIAct->setEnabled(!_on || dimension == 2);
		m_circle2DROIAct->setEnabled(!_on || dimension == 2);
		m_square2DROIAct->setEnabled(!_on || dimension == 2);
		m_polyline2DROIAct->setEnabled(!_on || dimension == 2);
	}

	//if (dimension == 2) return;
	if (act == m_xyAct) {
		m_line2DROIAct->setEnabled(true);// _on);
		m_triangle2DROIAct->setEnabled(dimension == 2);
		m_circle2DROIAct->setEnabled(dimension == 2);
		m_square2DROIAct->setEnabled(true);
		m_polyline2DROIAct->setEnabled(dimension == 2);
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
		m_line2DROIAct->setEnabled(true);// false);
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

void MainWindow::createDesignDock()
{
	QVBoxLayout* layoutAll = new QVBoxLayout;
	layoutAll->setContentsMargins(0, 0, 0, 0);
	layoutAll->setSpacing(6);
	layoutAll->addWidget(m_tabWidget);

	m_toolsPanel = new QWidget;
	m_toolsPanel->setObjectName("ToolsPanel");
	m_toolsPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_toolsPanel->setLayout(layoutAll);

	QScrollArea* toolsArea = new QScrollArea;
	toolsArea->setObjectName("ControlsScrollArea");
	toolsArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	toolsArea->setWidgetResizable(true);
	toolsArea->setFrameShape(QFrame::NoFrame);
	toolsArea->setWidget(m_toolsPanel);

	m_objectsTree = new QTreeWidget(this);
	m_objectsTree->setObjectName("ObjectsTree");
	m_objectsTree->setHeaderLabel(tr("Objects"));
	m_objectsTree->setAlternatingRowColors(false);
	m_objectsTree->setSelectionMode(QAbstractItemView::SingleSelection);
	m_objectsTree->setMinimumHeight(180);
	connect(m_objectsTree, &QTreeWidget::itemClicked, this, [this](QTreeWidgetItem* _item, int) {
		if (_item == NULL) return;
		const QVariant role = _item->data(0, Qt::UserRole);
		MyMultipleObject* multipleObject = (m_currentMdi != NULL && m_currentMdi->getWidget() != NULL)
			? dynamic_cast<MyMultipleObject*>(m_currentMdi->getWidget()->getObject()) : NULL;
		if (role.toString() == QStringLiteral("colorObject")) {
			bool ok = false;
			const int index = _item->data(0, Qt::UserRole + 1).toInt(&ok);
			if (!ok) return;
			if (multipleObject != NULL)
				multipleObject->setSelectedObjectIndices({ (size_t)index });
			changeColorObject(index);
			updateObjectsTreeSelectionCue(multipleObject);
			return;
		}
		if (role.toString() == QStringLiteral("hierarchyNode") && multipleObject != NULL) {
			bool ok = false;
			const int index = _item->data(0, Qt::UserRole + 1).toInt(&ok);
			if (!ok) return;
			multipleObject->setSelectedObjectIndices(multipleObject->collectObjectIndicesForHierarchyNode((size_t)index));
			if (multipleObject->hasSelectedObjectIndices())
				changeColorObject((int)multipleObject->selectedObjectIndices().front());
			updateObjectsTreeSelectionCue(multipleObject);
		}
	});

	m_applyAllObjectsButton = new QPushButton(tr("Apply to all objects"), this);
	m_recomputeGridButton = new QPushButton(tr("Recompute grid"), this);
	m_toggleGridCenteredButton = new QPushButton(tr("Toggle grid / centered"), this);
	m_exportObjectsButton = new QPushButton(tr("Export objects"), this);
	m_applyAllObjectsButton->setCheckable(true);
	m_toggleGridCenteredButton->setCheckable(true);
	connect(m_applyAllObjectsButton, &QPushButton::released, this, []() {
		poca::core::Engine::instance()->toggleGlobalCommands();
	});
	connect(m_recomputeGridButton, &QPushButton::released, this, &MainWindow::onGridReleased);
	connect(m_toggleGridCenteredButton, &QPushButton::clicked, this, &MainWindow::onToggleGridCentered);
	connect(m_exportObjectsButton, &QPushButton::released, this, &MainWindow::onExportAllObjects);

	QWidget* objectPanel = new QWidget(this);
	QVBoxLayout* objectLayout = new QVBoxLayout;
	objectLayout->setContentsMargins(0, 0, 0, 0);
	objectLayout->setSpacing(4);
	objectLayout->addWidget(m_objectsTree);
	QHBoxLayout* firstControlsRow = new QHBoxLayout;
	firstControlsRow->setContentsMargins(0, 0, 0, 0);
	firstControlsRow->setSpacing(4);
	firstControlsRow->addWidget(m_applyAllObjectsButton);
	firstControlsRow->addWidget(m_recomputeGridButton);
	objectLayout->addLayout(firstControlsRow);
	QHBoxLayout* secondControlsRow = new QHBoxLayout;
	secondControlsRow->setContentsMargins(0, 0, 0, 0);
	secondControlsRow->setSpacing(4);
	secondControlsRow->addWidget(m_toggleGridCenteredButton);
	secondControlsRow->addWidget(m_exportObjectsButton);
	objectLayout->addLayout(secondControlsRow);
	objectPanel->setLayout(objectLayout);

	m_objectTreeTabs = new QTabWidget(this);
	m_objectTreeTabs->setObjectName("ObjectsTreeTabs");
	configureInspectorTabWidget(m_objectTreeTabs);
	m_objectTreeTabs->addTab(objectPanel, tr("Objects"));
	m_objectTreeTabs->setVisible(false);

	m_propertiesTable = new QTableWidget(this);
	m_propertiesTable->setObjectName("PropertiesTable");
	m_propertiesTable->setColumnCount(2);
	m_propertiesTable->setHorizontalHeaderLabels(QStringList() << tr("Property") << tr("Value"));
	m_propertiesTable->horizontalHeader()->setStretchLastSection(true);
	m_propertiesTable->verticalHeader()->setVisible(false);
	m_propertiesTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
	m_propertiesTable->setSelectionMode(QAbstractItemView::NoSelection);
	m_propertiesTable->setFocusPolicy(Qt::NoFocus);

	m_leftInspectorTabs = new QTabWidget(this);
	m_leftInspectorTabs->setObjectName("LeftInspectorTabs");
	configureInspectorTabWidget(m_leftInspectorTabs);
	m_leftInspectorTabs->addTab(m_propertiesTable, tr("Properties"));
	m_leftInspectorTabs->addTab(toolsArea, tr("Controls"));

	QSplitter* objectsSplitter = new QSplitter(Qt::Vertical, this);
	objectsSplitter->setObjectName("LeftObjectsSplitter");
	objectsSplitter->addWidget(m_objectTreeTabs);
	objectsSplitter->addWidget(m_leftInspectorTabs);
	objectsSplitter->setStretchFactor(0, 1);
	objectsSplitter->setStretchFactor(1, 2);
	objectsSplitter->setChildrenCollapsible(false);

	QWidget* cameraPanel = new QWidget(this);
	QVBoxLayout* cameraLayout = new QVBoxLayout;
	cameraLayout->setContentsMargins(0, 0, 0, 0);
	cameraLayout->setSpacing(4);
	QGroupBox* cameraPathGroup = new QGroupBox(tr("Camera path"), cameraPanel);
	QVBoxLayout* cameraPathLayout = new QVBoxLayout;
	cameraPathLayout->setContentsMargins(4, 4, 4, 4);
	cameraPathLayout->addWidget(new QLabel(tr("Camera path controls are not available in this build."), cameraPathGroup));
	cameraPathGroup->setLayout(cameraPathLayout);
	cameraLayout->addWidget(cameraPathGroup);
	if (m_mfw != NULL && m_mfw->cameraPositionDockWidget() != NULL)
		cameraLayout->addWidget(m_mfw->cameraPositionDockWidget());
	if (m_mfw != NULL && m_mfw->generalDockWidget() != NULL)
		cameraLayout->addWidget(m_mfw->generalDockWidget());
	if (m_mfw != NULL && m_mfw->ssaoDockWidget() != NULL)
		cameraLayout->addWidget(m_mfw->ssaoDockWidget());
	QWidget* cameraSpacer = new QWidget(cameraPanel);
	cameraSpacer->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
	cameraLayout->addWidget(cameraSpacer);
	cameraPanel->setLayout(cameraLayout);

	QScrollArea* cameraArea = new QScrollArea;
	cameraArea->setObjectName("CameraScrollArea");
	cameraArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	cameraArea->setWidgetResizable(true);
	cameraArea->setFrameShape(QFrame::NoFrame);
	cameraArea->setWidget(cameraPanel);
	m_leftInspectorTabs->addTab(cameraArea, tr("Camera"));
	if (m_ROIsW != NULL)
		m_leftInspectorTabs->addTab(m_ROIsW, tr("ROI Manager"));
	poca::core::utils::processPendingWidgetsForNamedLayouts(this);

	m_objectsDockTabs = new QTabWidget(this);
	m_objectsDockTabs->setObjectName("ObjectsDockTabs");
	configureInspectorTabWidget(m_objectsDockTabs);
	if (m_macroW != NULL)
		m_objectsDockTabs->addTab(m_macroW, tr("Macro"));
	if (m_datasetAssemblerW != NULL)
		m_objectsDockTabs->addTab(m_datasetAssemblerW, tr("Assembler"));
	m_objectsDockObjectsPage = objectsSplitter;
	m_objectsDockTabs->addTab(m_objectsDockObjectsPage, tr("Objects"));

	m_designDock = new QDockWidget(tr("Objects"), this);
	m_designDock->setObjectName("ObjectsDock");
	m_designDock->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
	m_designDock->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_designDock->setMinimumWidth(300);
	m_designDock->setWidget(m_objectsDockTabs);
	addDockWidget(Qt::LeftDockWidgetArea, m_designDock);
}

void MainWindow::applyPrototypeStyle()
{
	menuBar()->setNativeMenuBar(false);
	m_mdiArea->setBackground(QBrush(QColor(236, 238, 240)));
	if (m_fileToolBar) {
		m_fileToolBar->setObjectName("MainPrototypeToolbar");
		m_fileToolBar->setMovable(false);
		m_fileToolBar->setIconSize(QSize(24, 24));
		m_fileToolBar->setToolButtonStyle(Qt::ToolButtonIconOnly);
	}
	setStyleSheet(R"(
		QMainWindow { background: #f2f4f6; }
		QMdiArea#MdiArea { background: #e9edf1; }
		QDockWidget#ObjectsDock { background: #f7f8fa; }
		QDockWidget#ObjectsDock::title { background: #f0f2f5; text-align: left; padding: 4px 6px; border-bottom: 1px solid #c9cdd2; }
		QTreeWidget#ObjectsTree, QTableWidget#PropertiesTable { background: #ffffff; color: #202124; gridline-color: #d7dbe0; border: 1px solid #c9cdd2; }
		QTableWidget#PropertiesTable QHeaderView::section { background: #f0f2f5; color: #202124; border: 1px solid #c9cdd2; padding: 3px; }
		QTreeWidget#ObjectsTree::item, QTableWidget#PropertiesTable::item { min-height: 20px; padding: 1px 4px; }
		QTreeWidget#ObjectsTree::item:selected { background: #dcecff; color: #111111; }
		QMdiArea#MdiArea { background: #eceef0; }
	)");
}

void MainWindow::configureInspectorTabWidget(QTabWidget* _tabWidget)
{
	if (_tabWidget == NULL || _tabWidget->tabBar() == NULL)
		return;
	_tabWidget->tabBar()->setExpanding(false);
	_tabWidget->tabBar()->setMovable(false);
	_tabWidget->tabBar()->setUsesScrollButtons(false);
	_tabWidget->setElideMode(Qt::ElideRight);
}

void MainWindow::addObjectToTree(poca::core::MyObjectInterface* _object, QTreeWidgetItem* _parent)
{
	if (_object == NULL || _parent == NULL) return;

	for (size_t n = 0; n < _object->nbBasicComponents(); n++) {
		poca::core::BasicComponentInterface* component = _object->getBasicComponent(n);
		if (component == NULL) continue;
		QTreeWidgetItem* componentItem = new QTreeWidgetItem(_parent);
		componentItem->setText(0, QString::fromStdString(component->getName()));
		componentItem->setCheckState(0, Qt::Checked);
		componentItem->setData(0, Qt::UserRole, QStringLiteral("component"));

		poca::core::BasicComponentList* list = dynamic_cast<poca::core::BasicComponentList*>(component);
		if (list == NULL || list->nbComponents() <= 1) continue;
		for (size_t i = 0; i < list->nbComponents(); i++) {
			poca::core::BasicComponent* child = list->getComponent(i);
			if (child == NULL) continue;
			QTreeWidgetItem* childItem = new QTreeWidgetItem(componentItem);
			childItem->setText(0, QString::fromStdString(child->getName()));
			childItem->setCheckState(0, Qt::Checked);
			childItem->setData(0, Qt::UserRole, QStringLiteral("subcomponent"));
		}
	}
}

void MainWindow::addHierarchyNodeToTree(MyMultipleObject* _object, size_t _nodeIndex, QTreeWidgetItem* _parent)
{
	if (_object == NULL || _parent == NULL || _nodeIndex >= _object->hierarchy().size()) return;
	const auto& node = _object->hierarchy()[_nodeIndex];
	QString label = QString::fromStdString(node.label.empty() ? node.levelName : node.label);
	if (!node.levelName.empty() && node.levelName != node.label)
		label = QString::fromStdString(node.levelName + ": " + node.label);

	QTreeWidgetItem* nodeItem = new QTreeWidgetItem(_parent);
	nodeItem->setText(0, label);
	nodeItem->setCheckState(0, Qt::Checked);
	nodeItem->setData(0, Qt::UserRole, QStringLiteral("hierarchyNode"));
	nodeItem->setData(0, Qt::UserRole + 1, int(_nodeIndex));

	for (const size_t childIndex : node.children)
		addHierarchyNodeToTree(_object, childIndex, nodeItem);

	for (const size_t objectIndex : node.objectIndices) {
		if (objectIndex >= _object->nbColors()) continue;
		poca::core::MyObjectInterface* childObject = _object->getObject(objectIndex);
		if (childObject == NULL) continue;
		QString childLabel = QString::fromStdString(childObject->getName().empty() ? std::string("Object") : childObject->getName());
		if (objectIndex == _object->currentObjectID())
			childLabel.append(tr("  [current]"));
		QTreeWidgetItem* objectItem = new QTreeWidgetItem(nodeItem);
		objectItem->setText(0, childLabel);
		objectItem->setCheckState(0, Qt::Checked);
		objectItem->setData(0, Qt::UserRole, QStringLiteral("colorObject"));
		objectItem->setData(0, Qt::UserRole + 1, int(objectIndex));
		addObjectToTree(childObject, objectItem);
	}
}

void MainWindow::updateObjectsTreeSelectionCue(MyMultipleObject* _object)
{
	if (m_objectsTree == NULL || _object == NULL) return;
	const std::vector<size_t>& selected = _object->selectedObjectIndices();
	std::function<void(QTreeWidgetItem*)> updateItem = [&](QTreeWidgetItem* item) {
		if (item == NULL) return;
		const QString role = item->data(0, Qt::UserRole).toString();
		bool isSelected = false;
		if (role == QStringLiteral("colorObject")) {
			bool ok = false;
			const int index = item->data(0, Qt::UserRole + 1).toInt(&ok);
			isSelected = ok && std::find(selected.begin(), selected.end(), (size_t)index) != selected.end();
		}
		else if (role == QStringLiteral("hierarchyNode")) {
			bool ok = false;
			const int index = item->data(0, Qt::UserRole + 1).toInt(&ok);
			if (ok) {
				const std::vector<size_t> nodeSelection = _object->collectObjectIndicesForHierarchyNode((size_t)index);
				isSelected = !nodeSelection.empty() && std::all_of(nodeSelection.begin(), nodeSelection.end(), [&](size_t objectIndex) {
					return std::find(selected.begin(), selected.end(), objectIndex) != selected.end();
				});
			}
		}
		item->setForeground(0, isSelected ? QBrush(QColor(0, 96, 180)) : QBrush(QColor(32, 33, 36)));
		QFont font = item->font(0);
		font.setBold(isSelected);
		item->setFont(0, font);
		for (int i = 0; i < item->childCount(); ++i)
			updateItem(item->child(i));
	};
	for (int i = 0; i < m_objectsTree->topLevelItemCount(); ++i)
		updateItem(m_objectsTree->topLevelItem(i));
}

void MainWindow::refreshObjectsPanel()
{
	if (m_objectsTree == NULL) return;
	m_objectsTree->clear();

	if (m_currentMdi == NULL || m_currentMdi->getWidget() == NULL || m_currentMdi->getWidget()->getObject() == NULL) {
		if (m_objectTreeTabs != NULL)
			m_objectTreeTabs->setVisible(false);
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
		if (m_objectsDockTabs != NULL && m_objectsDockObjectsPage != NULL)
			m_objectsDockTabs->setTabVisible(m_objectsDockTabs->indexOf(m_objectsDockObjectsPage), false);
#endif
		QTreeWidgetItem* emptyItem = new QTreeWidgetItem(m_objectsTree);
		emptyItem->setText(0, tr("No object loaded"));
		emptyItem->setFlags(emptyItem->flags() & ~Qt::ItemIsUserCheckable);
		m_objectsTree->expandAll();
		if (m_statusObjectLabel) m_statusObjectLabel->setText(tr("Objects: 0"));
		refreshObjectControls();
		return;
	}

	if (m_objectTreeTabs != NULL)
		m_objectTreeTabs->setVisible(true);
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	if (m_objectsDockTabs != NULL && m_objectsDockObjectsPage != NULL)
		m_objectsDockTabs->setTabVisible(m_objectsDockTabs->indexOf(m_objectsDockObjectsPage), true);
#endif

	poca::core::MyObjectInterface* object = m_currentMdi->getWidget()->getObject();
	QTreeWidgetItem* root = new QTreeWidgetItem(m_objectsTree);
	root->setText(0, QString::fromStdString(object->getName().empty() ? std::string("Object") : object->getName()));
	root->setCheckState(0, Qt::Checked);
	root->setData(0, Qt::UserRole, QStringLiteral("object"));

	MyMultipleObject* multipleObject = dynamic_cast<MyMultipleObject*>(object);
	if (multipleObject != NULL && multipleObject->hasHierarchy()) {
		std::vector<bool> attached(multipleObject->nbColors(), false);
		for (size_t n = 0; n < multipleObject->hierarchy().size(); n++)
			if (multipleObject->hierarchy()[n].parentIndex < 0)
				addHierarchyNodeToTree(multipleObject, n, root);
		for (const auto& node : multipleObject->hierarchy())
			for (const size_t objectIndex : node.objectIndices)
				if (objectIndex < attached.size())
					attached[objectIndex] = true;
		for (size_t n = 0; n < multipleObject->nbColors(); n++) {
			if (attached[n]) continue;
			poca::core::MyObjectInterface* childObject = multipleObject->getObject(n);
			if (childObject == NULL) continue;
			QString label = QString::fromStdString(childObject->getName().empty() ? std::string("Object") : childObject->getName());
			if (n == multipleObject->currentObjectID())
				label.append(tr("  [current]"));
			QTreeWidgetItem* objectItem = new QTreeWidgetItem(root);
			objectItem->setText(0, label);
			objectItem->setCheckState(0, Qt::Checked);
			objectItem->setData(0, Qt::UserRole, QStringLiteral("colorObject"));
			objectItem->setData(0, Qt::UserRole + 1, int(n));
			addObjectToTree(childObject, objectItem);
		}
	}
	else if (object->nbColors() > 1) {
		for (size_t n = 0; n < object->nbColors(); n++) {
			poca::core::MyObjectInterface* childObject = object->getObject(n);
			if (childObject == NULL) continue;
			QString label = QString::fromStdString(childObject->getName().empty() ? std::string("Object") : childObject->getName());
			if (n == object->currentObjectID())
				label.append(tr("  [current]"));
			QTreeWidgetItem* objectItem = new QTreeWidgetItem(root);
			objectItem->setText(0, label);
			objectItem->setCheckState(0, Qt::Checked);
			objectItem->setData(0, Qt::UserRole, QStringLiteral("colorObject"));
			objectItem->setData(0, Qt::UserRole + 1, int(n));
			if (n == object->currentObjectID())
				m_objectsTree->setCurrentItem(objectItem);
			addObjectToTree(childObject, objectItem);
		}
	}
	else
		addObjectToTree(object, root);

	updateObjectsTreeSelectionCue(multipleObject);
	m_objectsTree->expandToDepth(1);
	if (m_statusObjectLabel)
		m_statusObjectLabel->setText(tr("Objects: %1").arg(object->nbColors()));
	refreshObjectControls();
}

void MainWindow::refreshObjectControls()
{
	const bool hasMultipleObject = m_currentMdi != NULL
		&& m_currentMdi->getWidget() != NULL
		&& dynamic_cast<MyMultipleObject*>(m_currentMdi->getWidget()->getObject()) != NULL;
	if (m_applyAllObjectsButton) m_applyAllObjectsButton->setEnabled(hasMultipleObject);
	if (m_recomputeGridButton) m_recomputeGridButton->setEnabled(hasMultipleObject);
	if (m_toggleGridCenteredButton) m_toggleGridCenteredButton->setEnabled(hasMultipleObject);
	if (m_exportObjectsButton) m_exportObjectsButton->setEnabled(hasMultipleObject);
}

void MainWindow::refreshPropertiesPanel()
{
	if (m_propertiesTable == NULL) return;
	m_propertiesTable->setRowCount(0);

	auto addRow = [this](const QString& _property, const QString& _value) {
		const int row = m_propertiesTable->rowCount();
		m_propertiesTable->insertRow(row);
		m_propertiesTable->setItem(row, 0, new QTableWidgetItem(_property));
		m_propertiesTable->setItem(row, 1, new QTableWidgetItem(_value));
	};

	if (m_currentMdi == NULL || m_currentMdi->getWidget() == NULL || m_currentMdi->getWidget()->getObject() == NULL) {
		addRow(tr("State"), tr("No object loaded"));
		if (m_statusBoxLabel) m_statusBoxLabel->setText(tr("Box: -"));
		return;
	}

	poca::core::MyObjectInterface* object = m_currentMdi->getWidget()->getObject();
	const poca::core::BoundingBox bbox = object->boundingBox();

	addRow(tr("Name"), QString::fromStdString(object->getName()));
	addRow(tr("Directory"), QString::fromStdString(object->getDir()));
	addRow(tr("Dimension"), QString::number(object->dimension()));
	addRow(tr("Current object"), QString("%1 / %2").arg(object->currentObjectID() + 1).arg(object->nbColors()));
	addRow(tr("Basic components"), QString::number(object->nbBasicComponents()));
	addRow(tr("Box min"), QString("%1, %2, %3").arg(bbox[0], 0, 'f', 3).arg(bbox[1], 0, 'f', 3).arg(bbox[2], 0, 'f', 3));
	addRow(tr("Box max"), QString("%1, %2, %3").arg(bbox[3], 0, 'f', 3).arg(bbox[4], 0, 'f', 3).arg(bbox[5], 0, 'f', 3));
	addRow(tr("Box size"), QString("%1 x %2 x %3").arg(object->getWidth(), 0, 'f', 3).arg(object->getHeight(), 0, 'f', 3).arg(object->getThick(), 0, 'f', 3));

	for (size_t n = 0; n < object->nbBasicComponents(); n++) {
		poca::core::BasicComponentInterface* component = object->getBasicComponent(n);
		if (component == NULL) continue;
		addRow(tr("Component %1").arg(n + 1), QString::fromStdString(component->getName()));
	}

	m_propertiesTable->resizeColumnsToContents();
	if (m_statusBoxLabel)
		m_statusBoxLabel->setText(tr("Box: %1 x %2 x %3").arg(object->getWidth(), 0, 'f', 2).arg(object->getHeight(), 0, 'f', 2).arg(object->getThick(), 0, 'f', 2));
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
			poca::core::CommandExecutionContext context;
			poca::core::CommandExecutionResult result;
			engine->getPlugins()->execute(&ci, context, result);
			if (!result.has<poca::core::CreatedObjectContext>()) continue;
			poca::core::MyObjectInterface* obj = result.get<poca::core::CreatedObjectContext>().object;
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
	QObject::connect(cam, SIGNAL(objectCreated(poca::core::MyObjectInterface*)), this, SLOT(createWidget(poca::core::MyObjectInterface*)));
	m_mdiArea->addSubWindow(child);
	setActiveMdiChild(child);
	child->layout()->update();
	child->layout()->activate();
	child->show();

	_obj->notify("LoadObjCharacteristicsAllWidgets");

	engine->addCameraToObject(_obj, cam);
	cam->update();
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

	QString path = obj->getDir().c_str();// QDir::currentPath();
	QString filename = QFileDialog::getOpenFileName(0,
		QObject::tr("Select one component to add"),
		path,
		QObject::tr("Component files (*.*)"), 0);//, QFileDialog::DontUseNativeDialog | QFileDialog::DontUseCustomDirectoryIcons);

	if (filename.isEmpty()) return;

	execute(&poca::core::CommandInfo(true, "add", "filename", filename.toStdString()));
}

void MainWindow::addComponentToCurrentMdi(const QString& _filename)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	poca::core::MyObjectInterface* obj = cam->getObject();
	if (obj == NULL) return;
	
	std::cout << "Filename " << _filename.toStdString() << std::endl;
	poca::core::CommandInfo ci(false, "open", "path", std::string(_filename.toStdString()));

	poca::core::Engine* engine = poca::core::Engine::instance();
	if (engine->loadDataAndAddToObject(_filename, obj, &ci)) {
		poca::core::CommandInfo ci(false, "createDisplay");
		obj->executeGlobalCommand(&ci);
		obj->notify("LoadObjCharacteristicsAllWidgets");
		obj->notifyAll("updateDisplay");
	}
}

void MainWindow::openFile(const QString& _filename, poca::core::CommandInfo* _command)
{
	std::cout << __LINE__ << _filename.toStdString() << std::endl;
	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::PluginList* plugins = engine->getPlugins();
	poca::core::MyObjectInterface* obj = engine->loadDataAndCreateObject(_filename, _command);
	if (obj == NULL)
		return;
	poca::opengl::CameraInterface* cam = createWindows(obj);
	engine->addCameraToObject(obj, cam);
	cam->makeCurrent();
	poca::core::CommandInfo ci(false, "createDisplay");
	obj->executeGlobalCommand(&ci);
	std::cout << __LINE__ << _filename.toStdString() << std::endl;
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

	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::MyObjectInterface * newObj = engine->createObject(newName.toStdString(), dir, newDset);
	poca::opengl::CameraInterface* newCam = createWindows(newObj);
	engine->addCameraToObject(newObj, newCam);
	//createWindows(newDset, QString(dir.c_str()), newName);
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
		obj->attach(this, "LoadObjCharacteristicsAllWidgets");
		obj->attach(this, "UpdateMainTabWidgets");
		obj->attach(this, "duplicateCleanedData");

		MdiChild* child = new MdiChild(cam);
		QObject::connect(child, SIGNAL(setCurrentMdi(MdiChild*)), this, SLOT(setActiveMdiChild(MdiChild*)));
		QObject::connect(cam, SIGNAL(askForMovieCreation()), this, SLOT(createMovie()));
		QObject::connect(cam, SIGNAL(objectCreated(poca::core::MyObjectInterface*)), this, SLOT(createWidget(poca::core::MyObjectInterface*)));
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
		QObject::connect(cam, SIGNAL(objectCreated(poca::core::MyObjectInterface*)), this, SLOT(createWidget(poca::core::MyObjectInterface*)));
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
		/*QHBoxLayout* layout = NULL;
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
		m_widgetColors->updateGeometry();*/
		m_widgetColors->setCount(int(wobj->nbColors()));
		m_widgetColors->setCurrentIndex(int(wobj->currentObjectID()));
		m_widgetColors->hide();

		size_t dimension = wobj->dimension();
		m_line2DROIAct->setEnabled(true);// dimension == 2);
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
		
		//m_xyAct->setCheckable(dimension == 3);
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
	refreshObjectsPanel();
	refreshPropertiesPanel();
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
	foreach(QMdiSubWindow * window, m_mdiArea->subWindowList()) {
		MdiChild * mdiChild = qobject_cast <MdiChild *>(window);
		if (mdiChild != NULL)
			mdiChild->freeGPUResources();
	}
	m_mdiArea->closeAllSubWindows();
	m_currentMdi = NULL;
	refreshObjectsPanel();
	refreshPropertiesPanel();
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

	if (_action == "LoadObjCharacteristicsAllWidgets" || _action == "UpdateMainTabWidgets") {
		m_widgetColors->setCount(int(obj->nbColors()));
		m_widgetColors->setCurrentIndex(int(obj->currentObjectID()));
		m_widgetColors->hide();
		updateTabWidget();
		refreshObjectsPanel();
		refreshPropertiesPanel();
	}
	if (_action == "addCommandLastAddedComponent") {
		poca::core::BasicComponentInterface* bci = obj->getLastAddedBasicComponent();
		if (bci == NULL) return;
		plugins->addCommands(bci);

		poca::core::Engine::instance()->getObject(obj)->notify("LoadObjCharacteristicsAllWidgets");

		updateTabWidget();
		refreshObjectsPanel();
		refreshPropertiesPanel();
		obj->notify("LoadObjCharacteristicsAllWidgets");
	}
	if (_action == "duplicateCleanedData") {
		poca::core::CommandInfo ci(false, "getCleanedData");
		poca::core::CommandExecutionContext context;
		poca::core::CommandExecutionResult result;
		obj->executeCommandOnSpecificComponent("DetectionSet", &ci, context, result);
		if (result.has<poca::geometry::CleanedDetectionSetContext>()) {
			poca::geometry::DetectionSet* dset = result.get<poca::geometry::CleanedDetectionSetContext>().dset;
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
	m_currentMdi->resetViewer();
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

	//ColocalizationChoiceDialog* dial = new ColocalizationChoiceDialog(datasets);
	MergeDatasetsChoiceDialog* dial = new MergeDatasetsChoiceDialog(datasets);
	dial->setModal(true);
	if (dial->exec() == QDialog::Accepted) {
		std::vector < MdiChild*> objects = dial->getObjects();

		computeColocalization(objects, dial->batchComponentRendering());
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

void MainWindow::computeColocalization(const std::vector < MdiChild*>& _ws, const bool _batchComponentRendering)
{
	std::vector<poca::core::MyObjectInterface*> objs;
	for (MdiChild* mc : _ws)
		objs.push_back(mc->getWidget()->getObject());

	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::PluginList* plugins = engine->getPlugins();

	poca::core::MyObjectInterface* wobj = engine->generateMultipleObject(objs, _batchComponentRendering);
	if (wobj == NULL) return;

	MyMultipleObject* multiples = static_cast <MyMultipleObject*>(wobj);

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

		createWidget(multiples);

		/*poca::opengl::Camera* cam = new poca::opengl::Camera(wobj, wobj->dimension(), this);
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
		updateTabWidget();*/
	}
}

void MainWindow::changeColorObject(int _index)
{
	if (m_currentMdi == NULL) return;
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_currentMdi->getWidget());
	if (cam == NULL) return;
	poca::core::MyObjectInterface* obj = cam->getObject();
	if (obj == NULL) return;

	//int index = m_colorButtonsGroup->id(_button);
	obj->setCurrentObject(_index);
	obj->notify("LoadObjCharacteristicsAllWidgets");
	obj->notifyAll("updateDisplay");
	refreshObjectsPanel();
	refreshPropertiesPanel();
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
	/*if (m_currentMdi == NULL) {
		for (int n = 0; n < m_tabWidget->count(); n++)
			m_tabWidget->setTabVisible(n, m_tabWidget->tabText(n) == "Misc." || m_tabWidget->tabText(n) == "Macro");
	}*/
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
	else if (_com->nameCommand == "add") {
		if (m_currentMdi == NULL)
			return;
		QString filename;
		if (_com->hasParameter("filename"))
			filename = (_com->getParameter<std::string>("filename")).c_str();
		else {
			poca::core::MyObjectInterface* obj = m_currentMdi->getWidget()->getObject();
			QString dir = obj->getDir().c_str();
			QString name = obj->getName().c_str();
			if (!dir.endsWith('/'))
				dir.append("/");
			if (_com->hasParameter("appendToDir")) {
				std::string addToDir = _com->getParameter<std::string>("appendToDir");
				dir = dir + addToDir.c_str();
				if (!dir.endsWith('/'))
					dir.append("/");
			}
			if (_com->hasParameter("appendToName")) {
				std::string addToFile = _com->getParameter<std::string>("appendToName");
				int dotIndex = name.lastIndexOf('.');

				if (dotIndex != -1)
					name.insert(dotIndex, addToFile.c_str());
				else
					name.append(addToFile.c_str());
			}

			if (_com->hasParameter("extension")) {
				std::string ext = _com->getParameter<std::string>("extension");
				name.append(ext.c_str());
			}

			if (_com->hasParameter("replace")) {
				std::vector < std::vector<std::string>> vals = _com->getParameter< std::vector<std::vector<std::string>>>("replace");
				for(const auto& vec: vals)
					if (vec.size() == 2) {
						name.replace(vec[0].c_str(), vec[1].c_str());
					}
			}
			filename = dir + name;
		}
		addComponentToCurrentMdi(filename);
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

void MainWindow::openDatasetAssembler()
{
	if (m_objectsDockTabs == NULL || m_datasetAssemblerW == NULL) return;
	m_objectsDockTabs->setCurrentWidget(m_datasetAssemblerW);
	if (m_designDock != NULL) {
		m_designDock->show();
		m_designDock->raise();
	}
}

void MainWindow::openPythonWidget()
{
#ifndef NO_PYTHON
	if (m_pythonW == NULL) return;
	m_pythonW->show();
	m_pythonW->raise();
	m_pythonW->activateWindow();
#endif
}

void MainWindow::runMacro(std::vector<nlohmann::json> _macro, bool _onAllOpenedFiles)
{
	std::vector<MdiChild*> mdis;
	if (_onAllOpenedFiles) {
		QList<QMdiSubWindow*> windows = m_mdiArea->subWindowList();
		for (auto window : windows) {
			mdis.push_back(static_cast <MdiChild*>(window));
		}
	}
	else
		mdis.push_back(m_currentMdi);
	for (auto currentMdi : mdis) {
		setActiveMdiChild(currentMdi);
		for (auto json : _macro) {
			if (json.empty()) continue;

			const auto nameComp = json.begin().key();
			if (nameComp == "MainWindow") {
				runMacro(json[nameComp]);
				if (m_currentMdi != NULL && (currentMdi == NULL || currentMdi != m_currentMdi) )
					currentMdi = m_currentMdi;
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
				if (currentMdi == NULL) continue;
				poca::core::CommandableObject* comObj = NULL;
				poca::core::MyObjectInterface* obj = currentMdi->getWidget()->getObject();
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
							poca::core::CommandExecutionContext context;
							poca::core::CommandExecutionResult result;
							comObj->executeCommand(&command, context, result);
							if (result.has<poca::voronoi::CreatedDetectionSetContext>()) {
								poca::geometry::DetectionSet* dset = result.get<poca::voronoi::CreatedDetectionSetContext>().dset;
								if (dset == NULL) return;
								poca::geometry::DetectionSet* newDset = dset->duplicateSelection();
								const std::string& dir = obj->getDir(), name = obj->getName();
								QString newName(name.c_str());
								int index = newName.lastIndexOf(".");
								newName.insert(index, QString("_%1").arg(m_currentDuplicate++));
								createWindows(newDset, QString(dir.c_str()), newName);
							}
							else if (result.has<poca::core::CreatedObjectContext>()) {
								poca::core::MyObjectInterface* obj = result.get<poca::core::CreatedObjectContext>().object;
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
		if (currentMdi != NULL) {
			currentMdi->getWidget()->getObject()->notify("updateDisplay");
			currentMdi->getWidget()->getObject()->notify("LoadObjCharacteristicsAllWidgets");
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
							poca::core::CommandExecutionContext context;
							poca::core::CommandExecutionResult result;
							comObj->executeCommand(&command, context, result);
							if (result.has<poca::core::CreatedObjectContext>()) {
								poca::core::MyObjectInterface* obj = result.get<poca::core::CreatedObjectContext>().object;
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
	if (m_currentMdi != NULL) {
		m_currentMdi->getWidget()->getObject()->notify("updateDisplay");
		m_currentMdi->getWidget()->getObject()->notify("LoadObjCharacteristicsAllWidgets");
	}
}

void MainWindow::runMacro(const nlohmann::json& _json)
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	
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
	else if (tmp == "add") {
		poca::core::CommandInfo command(false, tmp);
		if(_json[tmp].contains("filename"))
			command.addParameter("filename", _json[tmp]["filename"].get<std::string>());
		if (_json[tmp].contains("appendToDir"))
			command.addParameter("appendToDir", _json[tmp]["appendToDir"].get<std::string>());
		if (_json[tmp].contains("appendToName"))
			command.addParameter("appendToName", _json[tmp]["appendToName"].get<std::string>());
		if (_json[tmp].contains("extension"))
			command.addParameter("extension", _json[tmp]["extension"].get<std::string>());
		if (_json[tmp].contains("replace"))
			command.addParameter("replace", _json[tmp]["replace"].get< std::vector<std::vector<std::string>>>());
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
			engine->executeCommand(voro, &poca::core::CommandInfo(false, "densityFactor", "factor", 1.6f));
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

	else if (tmp == "mergeLocalizationsDatasets") {
		std::vector <float> allxs, allys, allzs, allts;
		float count = 1;
		QString dir;
		for (auto window : m_mdiArea->subWindowList()) {
			MdiChild* child = qobject_cast <MdiChild*>(window);
			poca::core::MyObjectInterface* object = child->getWidget()->getObject();
			if (!object) continue;
			object = object->currentObject();
			if (!object) continue;
			if (!object->hasBasicComponent("DetectionSet")) continue;
			dir = object->getDir().c_str();
			poca::geometry::DetectionSet* dset = dynamic_cast<poca::geometry::DetectionSet*>(object->getBasicComponent("DetectionSet"));
			const std::vector <float>& xs = dset->getData<float>("x");
			const std::vector <float>& ys = dset->getData<float>("y");
			const std::vector <float>& zs = dset->hasData("z") ? dset->getData<float>("z") : std::vector <float>(xs.size(), 0.f);
			std::vector <float> ts = std::vector <float>(xs.size(), count);

			std::copy(xs.begin(), xs.end(), std::back_inserter(allxs));
			std::copy(ys.begin(), ys.end(), std::back_inserter(allys));
			std::copy(zs.begin(), zs.end(), std::back_inserter(allzs));
			std::copy(ts.begin(), ts.end(), std::back_inserter(allts));
			count++;
		}
		std::map <std::string, std::vector <float>> features;

		features["x"] = allxs;
		features["y"] = allys;
		features["z"] = allzs;
		features["frame"] = allts;

		poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(features);

		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::PluginList* plugins = engine->getPlugins();
		poca::core::MyObjectInterface* obj = engine->createObject(dir.toStdString(), "merged.csv", dset);
		if (obj == NULL)
			return;
		poca::opengl::CameraInterface* cam = createWindows(obj);
		engine->addCameraToObject(obj, cam);
	}
	else if (tmp == "colorForBacteries") {
		poca::core::Palette palette = poca::core::Palette::getStaticLut("HotCold2");
		poca::core::MyObjectInterface* object = m_currentMdi->getWidget()->getObject();
		const std::string& dir = object->getDir();
		std::ofstream fs(dir + std::string("/allStats.txt"));
		float nbs = (float)object->nbColors();
		std::vector <float> nbsCentroids, surfaces;
		for (auto n = 0; n < object->nbColors(); n++) {
			poca::core::Color4uc color = palette.getColorLUT((float)n / nbs);
			
			poca::core::MyObjectInterface* curObj = object->getObject(n);
			if (curObj->hasBasicComponent("DetectionSet")) {
				poca::core::BasicComponentInterface* bci = curObj->getBasicComponent("DetectionSet");
				bci->setPalette(new poca::core::Palette(color, color, "RandomOneColor"));
				engine->executeCommand(bci, false, "changeLUT");
			}
			if (curObj->hasBasicComponent("ObjectLists")) {
				poca::core::BasicComponentInterface* bci = curObj->getBasicComponent("ObjectLists");
				poca::core::BasicComponentList* bclist = static_cast <poca::core::BasicComponentList*>(bci);
				poca::core::BasicComponent* bc = bclist->currentComponent();
				bc->setPalette(new poca::core::Palette(color, color, "RandomOneColor"));
				engine->executeCommand(bc, false, "changeLUT");

				const std::vector <float>& nbs = bc->getData<float>("nbLocs");
				const std::vector <float>& area = bc->getData<float>("area");
				float totalArea = std::accumulate(area.begin(), area.end(), 0.f);
				float totalNbs = std::accumulate(nbs.begin(), nbs.end(), 0.f);

				fs << (n + 1) << "\t" << totalArea << "\t" << totalNbs << std::endl;
			}
		}
		fs.close();
	}
	else if (tmp == "organoidFeature") {
		std::ofstream fs("c:/tmp/values.txt");
		for (auto window : m_mdiArea->subWindowList()) {
			MdiChild* child = qobject_cast <MdiChild*>(window);
			poca::core::MyObjectInterface* object = child->getWidget()->getObject();
			if (!object) continue;
			object = object->currentObject();
			if (!object) continue;
			if (!object->hasBasicComponent("VoronoiDiagram")) continue;
			poca::geometry::VoronoiDiagram* vor = dynamic_cast<poca::geometry::VoronoiDiagram*>(object->getBasicComponent("VoronoiDiagram"));
			const std::vector <float>& ds = vor->getData<float>("meanDistance");
			for (const auto v : ds)
				fs << v << std::endl;
			/*if (!object->hasBasicComponent("ObjectLists")) continue;
			poca::geometry::ObjectLists* objects = dynamic_cast<poca::geometry::ObjectLists*>(object->getBasicComponent("ObjectLists"));
			auto nbs = objects->getObjectList(0)->nbObjects();
			const std::vector <float>& vols = objects->getObjectList(1)->getData<float>("volume");
			fs << object->getDir() << "\t" << object->getName() << "\t" << nbs << "\t" << vols[0] << std::endl;*/
		}
		fs.close();
	}
	else if (tmp == "checkAxialRatioOrganoids") {
		std::cout << "Starting checkAxialRatioOrganoids macro" << std::endl;
		poca::core::Engine* engine = poca::core::Engine::instance();
		engine->setVerbose(false);

		poca::core::PluginList* plugins = engine->getPlugins();
		float Pixel_Size = 0.3f; // Micrometres
		float Delta_Z = 1.f; // Micrometres
		float z_Ratio = Delta_Z / Pixel_Size;

		QString globalFolder("D:/Git/stardist-env/2025_03_03_ihssane_cycleGAN3D/");
		QStringList globalPaths;
		globalPaths << globalFolder + "20250217_IND_Incub24h/" << globalFolder + "20250227_Vc_Incub24h/" << globalFolder + "20251125_BCs_Maturation/7j/" << globalFolder + "20251125_BCs_Maturation/14j/" << globalFolder + "20251125_BCs_Maturation/21j/";

		std::ofstream fs(globalFolder.toStdString() + "axialRatio.txt");
		fs << "Folder\tCondition\tFile\tRadius\tVol ratio\t# frames\tAxial Ratio\tAcquired vol\tProj area\tEstimated Vol\tNorm vol organoid\tOrig vol organoid\n";

		for (const auto& globalPath : globalPaths) {
			QDir dirGlobal(globalPath);
			std::cout << dirGlobal.dirName().toStdString() << std::endl;
			QFileInfoList conditions = dirGlobal.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
			for (const auto& condition : conditions) {
				std::cout << condition.absoluteFilePath().toStdString() << std::endl;
				QStringList allFiles;
				QDir dir(condition.absoluteFilePath() + "/masks");
				QFileInfoList list = dir.entryInfoList(QDir::Files);
				for (int i = 0; i < list.size(); ++i) {
					QFileInfo dirInfo = list.at(i);
					QString fileName = dirInfo.fileName();
					allFiles << fileName;
					std::cout << fileName.toStdString() << std::endl;
					//if (dirName == "." || dirName == "..") continue;
				}

				for (auto currentFile : allFiles) {
					QString samName = currentFile, nucleiObjectsName = currentFile, centroidsName = currentFile;
					samName.replace("w2", "w4");
					samName.replace("405", "640");
					nucleiObjectsName.replace(".tif", ".obj");
					QString orgaObjectName = samName;
					orgaObjectName.replace(".tif", ".obj");
					centroidsName.replace(".tif", ".csv");
					//Determine if all files are present
					bool labelImageExist = QFileInfo::exists(condition.absoluteFilePath() + "/masks/" + currentFile);
					bool samImageExist = QFileInfo::exists(condition.absoluteFilePath() + "/SAM2_Actin_Mask/" + samName);
					if (!labelImageExist || !samImageExist) {
						fs << dirGlobal.dirName().toStdString() << "\t" << condition.fileName().toStdString() << "\t" << currentFile.toStdString() << "\t" "Failed with non existent file" << std::endl;
						std::cout << dirGlobal.dirName().toStdString() << "\t" << condition.fileName().toStdString() << "\t" << currentFile.toStdString() << "\t" "Failed with non existent file" << std::endl;
						continue;
					}

					std::cout << "Opening condition " << dirGlobal.dirName().toStdString() << "/" << condition.fileName().toStdString() << ", file " << currentFile.toStdString() << std::endl;

					poca::core::BasicComponentInterface* bci = engine->loadData(condition.absoluteFilePath() + "/SAM2_Actin_Mask/" + samName);
					if (bci == NULL) {
						fs << dirGlobal.dirName().toStdString() << "\t" << condition.fileName().toStdString() << "\t" << currentFile.toStdString() << "\t" "Failed to load actin file" << std::endl;
						std::cout << dirGlobal.dirName().toStdString() << "\t" << condition.fileName().toStdString() << "\t" << currentFile.toStdString() << "\t" "Failed to load actin file" << std::endl;
						continue;
					}

					poca::core::ImagesList* imlist = static_cast <poca::core::ImagesList*>(bci);

					poca::core::Image<uint8_t>* actin8bits = static_cast<poca::core::Image<uint8_t>*>(imlist->currentImage());
					if (actin8bits == NULL) {
						fs << dirGlobal.dirName().toStdString() << "\t" << condition.fileName().toStdString() << "\t" << currentFile.toStdString() << "\t" "Failed, Actin segmentation should be uint8_t" << std::endl;
						std::cout << dirGlobal.dirName().toStdString() << "\t" << condition.fileName().toStdString() << "\t" << currentFile.toStdString() << "\t" "Failed, Actin segmentation should be uint8_t" << std::endl;
						continue;
					}
					/*std::vector <uint8_t>& pixelOrig = actin8bits->pixels();

					thrust::device_vector<float> d_labels, d_counts;
					thrust::device_vector<float> d_pixels_stack(pixelOrig);
					count_occurences_label_kernel_gpu< float>(d_pixels_stack, d_labels, d_counts);
					std::vector <float> volumeActin(d_counts.size() - 1);
					cudaMemcpy(volumeActin.data(), thrust::raw_pointer_cast(d_counts.data() + 1), volumeActin.size() * sizeof(float), cudaMemcpyDeviceToHost);
					float volumesAcquired = volumeActin[0] * z_Ratio;

					std::vector <uint8_t> maxProj;
					maxProjection<uint8_t>(pixelOrig, maxProj, actin8bits->width(), actin8bits->height(), actin8bits->depth());
					thrust::device_vector<float> d_pixels(maxProj);
					count_occurences_label_kernel_gpu< float>(d_pixels, d_labels, d_counts);
					std::vector <float> surfaceProj(d_counts.size() - 1);
					cudaMemcpy(surfaceProj.data(), thrust::raw_pointer_cast(d_counts.data() + 1), surfaceProj.size() * sizeof(float), cudaMemcpyDeviceToHost);
					float surfaceAcquired = surfaceProj[0];*/

					float volumesAcquired = 0.f, surfaceAcquired = 0.f;
					const std::vector <uint8_t>& pixelOrig = actin8bits->pixels();
					std::vector<uint32_t> volumeActin, labelsActin;
					count_occurences_label(pixelOrig, labelsActin, volumeActin, 1);
					volumesAcquired = volumeActin[0] * z_Ratio;
					std::vector <uint8_t> maxProj;
					std::vector<uint32_t> surfaceProj;
					maxProjection<uint8_t>(pixelOrig, maxProj, actin8bits->width(), actin8bits->height(), actin8bits->depth());
					count_occurences_label(maxProj, labelsActin, surfaceProj, 1);
					surfaceAcquired = surfaceProj[0];

					float r = sqrt(surfaceAcquired / M_PI);
					float volSphere = (4.0 / 3.0) * M_PI * pow(r, 3);
					float volRatio = volumesAcquired / volSphere;
					float axialRatio = ((2 * r) * volRatio) / float(actin8bits->depth());

					fs << dirGlobal.dirName().toStdString() << "\t" << condition.fileName().toStdString() << "\t" << currentFile.toStdString() << "\t" << r << "\t" << volRatio << "\t" << actin8bits->depth() << "\t" << axialRatio <<
						"\t" << volumesAcquired << "\t" << surfaceAcquired << "\t" << volSphere << "\t" << (volumesAcquired / volRatio) << "\t" << volumeActin[0] << std::endl;
					std::cout << dirGlobal.dirName().toStdString() << ", " << condition.fileName().toStdString() << ", " << currentFile.toStdString() << ", " << r << ", " << volRatio << ", " << actin8bits->depth() << ", " << axialRatio <<
						", " << volumesAcquired << ", " << surfaceAcquired << ", " << volSphere << ", " << (volumesAcquired / volRatio) << ", " << volumeActin[0] << std::endl;

					//delete actin8bits;
					delete bci;
				}
			}
		}
		fs.close();
	}
	else if (tmp == "processOrganoids") {
		std::cout << "Starting processOrganoids macro" << std::endl;
		poca::core::Engine* engine = poca::core::Engine::instance();
		engine->setVerbose(false);

		poca::core::PluginList* plugins = engine->getPlugins();
		float axial_ration = 3.8f;

		QString globalFolder("D:/Git/stardist-env/2025_03_03_ihssane_cycleGAN3D/");
		QStringList globalPaths;
		globalPaths << globalFolder + "20250217_IND_Incub24h/" << globalFolder + "20250227_Vc_Incub24h/" << globalFolder + "20251125_BCs_Maturation/7j/" << globalFolder + "20251125_BCs_Maturation/14j/" << globalFolder + "20251125_BCs_Maturation/21j/";

		for (const auto& globalPath : globalPaths) {
			QDir dirGlobal(globalPath);
			std::cout << dirGlobal.dirName().toStdString() << std::endl;
			QFileInfoList conditions = dirGlobal.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
			for (const auto& condition : conditions) {
				std::cout << condition.absoluteFilePath().toStdString() << std::endl;
				QStringList allFiles;
				QDir dir(condition.absoluteFilePath() + "/masks");
				QFileInfoList list = dir.entryInfoList(QDir::Files);
				for (int i = 0; i < list.size(); ++i) {
					QFileInfo dirInfo = list.at(i);
					QString fileName = dirInfo.fileName();
					allFiles << fileName;
					std::cout << fileName.toStdString() << std::endl;
					//if (dirName == "." || dirName == "..") continue;
				}

				for (auto currentFile : allFiles) {
					QString samName = currentFile, nucleiObjectsName = currentFile, centroidsName = currentFile;
					samName.replace("w2", "w4");
					samName.replace("405", "640");
					nucleiObjectsName.replace(".tif", ".obj");
					QString orgaObjectName = samName;
					orgaObjectName.replace(".tif", ".obj");
					centroidsName.replace(".tif", ".csv");
					//Determine if all files are present
					bool labelImageExist = QFileInfo::exists(condition.absoluteFilePath() + "/masks/" + currentFile);
					bool samImageExist = QFileInfo::exists(condition.absoluteFilePath() + "/SAM2_Actin_Mask/" + samName);
					if (!labelImageExist || !samImageExist) {
						std::cout << dirGlobal.dirName().toStdString() << "\t" << condition.fileName().toStdString() << "\t" << currentFile.toStdString() << "\t" "Failed with non existent file" << std::endl;
						continue;
					}

					std::cout << "Opening condition " << dirGlobal.dirName().toStdString() << "/" << condition.fileName().toStdString() << ", file " << currentFile.toStdString() << std::endl;

					poca::core::BasicComponentInterface* bci = engine->loadData(condition.absoluteFilePath() + "/SAM2_Actin_Mask/" + samName);
					if (bci == NULL) {
						std::cout << dirGlobal.dirName().toStdString() << "\t" << condition.fileName().toStdString() << "\t" << currentFile.toStdString() << "\t" "Failed to load actin file" << std::endl;
						continue;
					}
					poca::core::BasicComponentInterface* bci2 = engine->loadData(condition.absoluteFilePath() + "/masks/" + currentFile);
					if (bci == NULL) {
						std::cout << dirGlobal.dirName().toStdString() << "\t" << condition.fileName().toStdString() << "\t" << currentFile.toStdString() << "\t" "Failed to load nuclei file" << std::endl;
						continue;
					}

					poca::core::ImagesList* imlist = dynamic_cast <poca::core::ImagesList *>(engine->mergeComponentLists(bci, bci2));
					plugins->addCommands(imlist);

					QDir dirFolders;
					if (!dirFolders.exists(condition.absoluteFilePath() + "/actinObjs2")) {
						if (!dir.mkdir(condition.absoluteFilePath() + "/actinObjs2")) {
							std::cout << "Failed to create " << condition.absoluteFilePath().toStdString() << std::endl;
							continue;
						}
					}
					if (!dirFolders.exists(condition.absoluteFilePath() + "/nucleusCentroids2")) {
						if (!dir.mkdir(condition.absoluteFilePath() + "/nucleusCentroids2")) {
							std::cout << "Failed to create " << condition.absoluteFilePath().toStdString() << std::endl;
							continue;
						}
					}
					if (!dirFolders.exists(condition.absoluteFilePath() + "/nucleusObjs2")) {
						if (!dir.mkdir(condition.absoluteFilePath() + "/nucleusObjs2")) {
							std::cout << "Failed to create " << condition.absoluteFilePath().toStdString() << std::endl;
							continue;
						}
					}
					if (!dirFolders.exists(condition.absoluteFilePath() + "/voronoiCut")) {
						if (!dir.mkdir(condition.absoluteFilePath() + "/voronoiCut")) {
							std::cout << "Failed to create " << condition.absoluteFilePath().toStdString() << std::endl;
							continue;
						}
					}

					std::cout << "Marching cubes actin" << std::endl;
					imlist->setCurrentComponentIndex(0);
					plugins->addCommands(imlist->getImage(0));
					poca::core::CommandInfo ci(false, "marchingCubes", "threshold", 0.5f, "repair", true, "remeshing", true, "targetLength", 6.f, "iterations", (uint32_t)3, "inROIs", false, "scaleZ", 3.8f);
					poca::core::CommandExecutionContext context;
					poca::core::CommandExecutionResult result;
					engine->executeCommand(imlist, &ci, context, result);
					if (!result.has<poca::geometry::CreatedObjectListMeshContext>()) {
						std::cout << "Failed to marching cube actin " << std::endl;
						continue;
					}
					std::cout << "Subdividing actin" << std::endl;
					poca::geometry::ObjectListMesh* actin = result.get<poca::geometry::CreatedObjectListMeshContext>().objects;
					plugins->addCommands(actin);
					ci = poca::core::CommandInfo(false, "subdivide", "iterations", (uint32_t)1);
					result = poca::core::CommandExecutionResult();
					engine->executeCommand(actin, &ci, context, result);
					if (!result.has<poca::geometry::CreatedObjectListMeshContext>()) {
						std::cout << "Failed to subdivide actin " << std::endl;
						continue;
					}
					poca::geometry::ObjectListMesh* actinSmooth = result.get<poca::geometry::CreatedObjectListMeshContext>().objects;
					
					std::cout << "Marching cubes nuclei" << std::endl;
					imlist->setCurrentComponentIndex(1);
					plugins->addCommands(imlist->getImage(1));
					ci = poca::core::CommandInfo(false, "marchingCubes", "threshold", 0.5f, "repair", true, "remeshing", true, "targetLength", 6.f, "iterations", (uint32_t)3, "inROIs", false, "scaleZ", 3.8f);
					result = poca::core::CommandExecutionResult();
					engine->executeCommand(imlist, &ci, context, result);
					if (!result.has<poca::geometry::CreatedObjectListMeshContext>()) {
						std::cout << "Failed to marching cube nuclei " << std::endl;
						continue;
					}
					std::cout << "Subdividing nuclei" << std::endl;
					poca::geometry::ObjectListMesh* noyaux = result.get<poca::geometry::CreatedObjectListMeshContext>().objects;
					plugins->addCommands(noyaux);
					ci = poca::core::CommandInfo(false, "subdivide", "iterations", (uint32_t)1);
					result = poca::core::CommandExecutionResult();
					engine->executeCommand(noyaux, &ci, context, result);
					if (!result.has<poca::geometry::CreatedObjectListMeshContext>()) {
						std::cout << "Failed to subdivide nuclei " << std::endl;
						continue;
					}
					poca::geometry::ObjectListMesh* noyauxSmooth = result.get<poca::geometry::CreatedObjectListMeshContext>().objects;
					//delete actin8bits;

					std::cout << "Creation of the nculei centroids" << std::endl;
					std::vector <poca::core::Vec3mf> centroids(noyauxSmooth->nbElements());
					for (size_t n = 0; n < noyauxSmooth->nbElements(); n++)
						centroids[n] = noyauxSmooth->computeBarycenterElement(n);
					std::vector <bool> selectedNuclei(noyauxSmooth->nbElements());
					const auto& actinMesh = actinSmooth->getMeshes()[0];
					CGAL::Side_of_triangle_mesh<Surface_mesh_3_double, Kernel> inside(actinMesh);
					for (auto n = 0; n < noyauxSmooth->nbElements(); n++)
						selectedNuclei[n] = poca::geometry::insideMesh(inside, centroids[n].x(), centroids[n].y(), centroids[n].z());
					//Create DetectionSet of the selected centroids
					std::map <std::string, std::vector <float>> features;
					std::vector <float> xs, ys, zs;
					for (size_t n = 0; n < centroids.size(); n++) {
						if (!selectedNuclei[n]) continue;
						xs.push_back(centroids[n][0]);
						ys.push_back(centroids[n][1]);
						zs.push_back(centroids[n][2]);
					}
					features["x"] = xs;
					features["y"] = ys;
					features["z"] = zs;
					std::map <std::string, poca::core::MyData*> featuresObjects = noyauxSmooth->getData();
					for (const auto& feature : featuresObjects) {
						if (feature.first != "x" && feature.first != "y" && feature.first != "z") {
							std::vector <float> selectedValues;
							const std::vector <float>& values = feature.second->getData<float>();
							for (size_t n = 0; n < centroids.size(); n++) {
								if (!selectedNuclei[n]) continue;
								selectedValues.push_back(values[n]);
							}
							features[feature.first] = selectedValues;
						}
					}
					poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(features);

					std::cout << "Filter nuclei wrt actin" << std::endl;
					const auto& nucleiMeshes = noyauxSmooth->getMeshes();
					std::vector <Surface_mesh_3_double> selectedMeshes;
					for (size_t n = 0; n < centroids.size(); n++) {
						if (!selectedNuclei[n]) continue;
						selectedMeshes.emplace_back(nucleiMeshes[n]);
					}
					poca::geometry::ObjectListMesh* noyauxSmoothFiltered = new poca::geometry::ObjectListMesh(selectedMeshes);

					std::cout << "Creation of the 3D Voronoi" << std::endl;
					poca::geometry::DelaunayTriangulationFactoryInterface* factory = poca::geometry::createDelaunayTriangulationFactory();
					poca::geometry::DelaunayTriangulationInterface* delaunay = factory->createDelaunayTriangulation(xs, ys, zs);
					poca::geometry::VoronoiDiagramFactoryInterface* factoryV = poca::geometry::createVoronoiDiagramFactory();
					poca::geometry::KdTree_DetectionPoint* kdtree = dset->getKdTree();
					poca::geometry::VoronoiDiagram* voronoi = factoryV->createVoronoiDiagram(xs, ys, zs, kdtree, delaunay, false);
					poca::geometry::VoronoiDiagram3D* voro3D = static_cast <poca::geometry::VoronoiDiagram3D*>(voronoi);
					const auto& polyhedrons = voro3D->getPolyhedrons();
					std::vector < Surface_mesh_3_double> voronoiCells, voronoiCellsRemeshed, voronoiCellsCut;
					std::cout << "Cutting of the 3D Voronoi cells" << std::endl;
					for (const auto& poly : polyhedrons) {
						voronoiCells.push_back(Surface_mesh_3_double());
						CGAL::copy_face_graph(poly, voronoiCells.back());
						assert(CGAL::is_valid_polygon_mesh(voronoiCells.back()));
					}
					//Remesh the voronoi cells
					std::cout << "Remeshing of the 3D Voronoi cells" << std::endl;
					for (const auto& cell : voronoiCells) {
						auto rcell = cell;
						auto eif = get(CGAL::edge_is_feature, rcell);
						auto pid = get(CGAL::face_patch_id_t<int>(), rcell);
						auto vip = get(CGAL::vertex_incident_patches_t<int>(), rcell);

						const double sharp_angle_deg = 2.0;

						// 1) Detect feature (sharp) edges
						CGAL::Polygon_mesh_processing::sharp_edges_segmentation(
							rcell, sharp_angle_deg, eif, pid,
							CGAL::parameters::vertex_incident_patches_map(vip)
						);

						// 2) Collect feature edges
						using edge_descriptor = boost::graph_traits<decltype(rcell)>::edge_descriptor;
						std::vector<edge_descriptor> feature_edges;
						feature_edges.reserve(num_edges(rcell));

						for (edge_descriptor e : edges(rcell))
							if (get(eif, e))
								feature_edges.push_back(e);

						// 3) Choose target length (IMPORTANT: use a sane value in your units)
						const double target_edge_length = 3/* e.g. 0.6 * avg edge length, not a hardcoded 3 */;
						const unsigned int nb_iter = 3;

						// 4) Split feature edges so they're already close to target
						CGAL::Polygon_mesh_processing::split_long_edges(feature_edges, target_edge_length, rcell);

						// (Optional but robust) re-run feature detection so newly created edges are marked too
						CGAL::Polygon_mesh_processing::sharp_edges_segmentation(
							rcell, sharp_angle_deg, eif, pid,
							CGAL::parameters::vertex_incident_patches_map(vip)
						);

						// 5) Now remesh, protecting constraints (feature polyline stays, but now it can have vertices)
						CGAL::Polygon_mesh_processing::isotropic_remeshing(
							faces(rcell),
							target_edge_length,
							rcell,
							CGAL::parameters::
							number_of_iterations(nb_iter).
							protect_constraints(true).
							edge_is_constrained_map(eif)
						);

						voronoiCellsRemeshed.push_back(rcell);
					}

					std::cout << "Cutting of the 3D Voronoi cells" << std::endl;
					poca::geometry::meshesInsideMeshWithCutting(actinMesh, voronoiCellsRemeshed, voronoiCellsCut);
					poca::geometry::ObjectListMesh* voronoiCellsCutMesh = new poca::geometry::ObjectListMesh(voronoiCellsCut);


					std::cout << "Saving" << std::endl;
					actinSmooth->saveAsOBJ((condition.absoluteFilePath() + "/actinObjs2/" + orgaObjectName).toStdString());
					noyauxSmoothFiltered->saveAsOBJ((condition.absoluteFilePath() + "/nucleusObjs2/" + nucleiObjectsName).toStdString());
					dset->saveDetections((condition.absoluteFilePath() + "/nucleusCentroids2/" + centroidsName).toStdString());
					voronoiCellsCutMesh->saveAsOBJ((condition.absoluteFilePath() + "/voronoiCut/" + nucleiObjectsName).toStdString());

					delete voronoiCellsCutMesh;
					delete factoryV;
					delete factory;
					delete delaunay;
					delete voronoi;
					delete dset;
					delete noyauxSmoothFiltered;
					delete noyauxSmooth;
					delete actinSmooth;
					delete noyaux;
					delete actin;
					delete bci;
				}
			}
		}
	}
	else if (tmp == "organoidFeature_2") {
		std::cout << "Starting organoidFeature_2 macro" << std::endl;
		poca::core::Engine* engine = poca::core::Engine::instance();
		engine->setVerbose(false);

		poca::core::PluginList* plugins = engine->getPlugins();
		float Pixel_Size = 0.3f; // Micrometres
		float Delta_Z = 1.f; // Micrometres
		float z_Ratio = Delta_Z / Pixel_Size;

		QString globalPath("D:/Git/stardist-env/2025_03_03_ihssane_cycleGAN3D/20250217_IND_Incub24h/");
		QStringList conditions;
		conditions << "Control/" << "10uM/" << "33uM/" << "100uM/" << "300uM/" << "600uM/";
		std::ofstream fs(globalPath.toStdString() + "values.txt");
		fs << "Condition\tFile\tAcquired vol\tProj area\tEstimated ratio\tEstimated Vol\tEstimated ratio\tNorm vol organoid\tNorm # nuclei\tOrig vol organoid\tOrig # nuclei" << std::endl;
		for (const auto& condition : conditions) {
			//Get names of files
			QStringList allFiles;
			QDir dir(globalPath + condition + "masks");
			dir.setFilter(QDir::Files);
			QFileInfoList list = dir.entryInfoList();
			for (int i = 0; i < list.size(); ++i) {
				QFileInfo dirInfo = list.at(i);
				QString fileName = dirInfo.fileName();
				allFiles << fileName;
				//std::cout << fileName.toStdString() << std::endl;
				//if (dirName == "." || dirName == "..") continue;
			}

			std::vector <float> volumesAcquired, surfaceAcquired, nbNuclei, volOrganoid;
			std::vector <std::string> names;
			for (auto currentFile : allFiles) {
				QString samName = currentFile, nucleiObjectsName = currentFile, centroidsName = currentFile;
				samName.replace("w2", "w4");
				samName.replace("405", "640");
				nucleiObjectsName.replace(".tif", ".obj");
				QString orgaObjectName = samName;
				orgaObjectName.replace(".tif", ".obj");
				centroidsName.replace(".tif", ".csv");
				//Determine if all files are present
				bool labelImageExist = QFileInfo::exists(globalPath + condition + "masks/" + currentFile);
				bool samImageExist = QFileInfo::exists(globalPath + condition + "SAM2_Actin_Mask/" + samName);
				bool nucleiObjsExist = QFileInfo::exists(globalPath + condition + "nucleusObjs/" + nucleiObjectsName);
				bool organoidObjExist = QFileInfo::exists(globalPath + condition + "actinObjs/" + orgaObjectName);
				bool centroidsExist = QFileInfo::exists(globalPath + condition + "nucleusCentroids/" + centroidsName);

				/*if (!labelImageExist || !samImageExist || !nucleiObjsExist || !organoidObjExist || !centroidsExist)
				{
					std::cout << currentFile.toStdString() << ", not good" << std::endl;
					continue;
				}*/

				//Open actin sam segmentation
				poca::core::CommandInfo ci;
				/*poca::core::MyObjectInterface* obj = engine->loadDataAndCreateObject(globalPath + condition + "SAM2_Actin_Mask/" + samName, &ci);
				if (obj == NULL || !obj->hasBasicComponent("ImagesList")) {
					std::cout << "Loading actin image failed" << std::endl;
					continue;
				}
				poca::core::ImagesList* imlist = static_cast <poca::core::ImagesList*>(obj->getBasicComponent("ImagesList"));
				poca::core::ImageInterface* actin = imlist->currentImage();
				engine->executeCommand(actin, false, "changeImageType", poca::core::LABEL);
				if (!actin->hasData("volume")) {
					std::cout << "Actin image has not volume" << std::endl;
					continue;
				}
				const std::vector <float>& volume = actin->getData<float>("volume");
				volumesAcquired.push_back(volume[0] * z_Ratio);*/

				std::cout << "Opening condition " << condition.toStdString() << ", file " << currentFile.toStdString() << std::endl;

				poca::core::BasicComponentInterface* bci = engine->loadData(globalPath + condition + "SAM2_Actin_Mask/" + samName);
				if (bci == NULL) {
					std::cout << "Loading actin image failed" << std::endl;
					continue;
				}

				poca::core::ImagesList* imlist = static_cast <poca::core::ImagesList*>(bci);

				poca::core::Image<uint8_t>* actin8bits = static_cast<poca::core::Image<uint8_t>*>(imlist->currentImage());
				if (actin8bits == NULL) {
					std::cout << "Actin segmentation should be uint8_t" << std::endl;
					continue;
				}
				std::vector <uint8_t>& pixelOrig = actin8bits->pixels();
				std::vector<uint32_t> volumeActin, labelsActin;
				count_occurences_label(pixelOrig, labelsActin, volumeActin, 1);
				volumesAcquired.push_back(volumeActin[0] * z_Ratio);

				std::vector <uint8_t> maxProj;
				std::vector<uint32_t> surfaceProj;
				maxProjection<uint8_t>(pixelOrig, maxProj, actin8bits->width(), actin8bits->height(), actin8bits->depth());
				count_occurences_label(maxProj, labelsActin, surfaceProj, 1);
				surfaceAcquired.push_back(surfaceProj[0]);

				bci = engine->loadData(globalPath + condition + "nucleusObjs/" + nucleiObjectsName);
				nbNuclei.push_back(bci->nbElements());
				delete bci;
				bci = engine->loadData(globalPath + condition + "actinObjs/" + orgaObjectName);
				const std::vector <float>& organoidVolumeObject = bci->getMyData("volume")->getData<float>();
				volOrganoid.push_back(organoidVolumeObject[0]);

				delete actin8bits;
				delete bci;

				names.push_back(currentFile.toStdString());
			}
			for (auto n = 0; n < volumesAcquired.size(); n++) {
				float r = sqrt(surfaceAcquired[n] / M_PI);
				float volSphere = (4.0 / 3.0) * M_PI * pow(r, 3);
				float volRatio = volumesAcquired[n] / volSphere;
				std::cout << "Acquired Volume = " << volumesAcquired[n] << ", Proj Area = " << surfaceAcquired[n] << ", Estimated Radius = " << r <<
					", Estimated Volume = " << volSphere << ", Estimated Volume Ratio = " << (volRatio * 100.f) << "%, Norm vol organoid = " << (volOrganoid[n] / volRatio) <<
					" (orig = " << volOrganoid[n] << "), Norm nb nuclei = " << int(nbNuclei[n] / volRatio) << " (orig = " << nbNuclei[n] << ")" << std::endl;
				fs << condition.toStdString() << "\t" << names[n] << "\t" << volumesAcquired[n] << "\t" << surfaceAcquired[n] << "\t" << r << "\t" << volSphere << "\t" <<
					(volRatio * 100.f) << "\t" << (volOrganoid[n] / volRatio) << "\t" << int(nbNuclei[n] / volRatio) << "\t" << volOrganoid[n] << "\t" << nbNuclei[n] << std::endl;
			}
		}
		fs.close();
	}
	else if (tmp == "allMasksSingapour") {
		std::cout << "Starting allMasksSingapour macro" << std::endl;
		poca::core::Engine* engine = poca::core::Engine::instance();
		engine->setVerbose(false);
		poca::core::PluginList* plugins = engine->getPlugins();

		QStringList paths;
		//paths << "D:/Git/stardist-env/2021_05_31_trainingFluo/Round5-D10-ScreenCentri3/masks/";
		//paths << "D:/Git/stardist-env/2021_05_31_trainingFluo/Round5-D10-ScreenCentri-Membrane/masks/";
		paths << "D:/DataPALMSTORM/nucleus_segmentation/all_masks_singapour/indiv/";
		std::vector <Surface_mesh_3_double> meshes;
		std::vector <poca::core::BoundingBox> bboxes;
		std::vector<stbrp_rect> rects;
		int cur = 0;
		uint32_t PADDING_BINS = 2;

		for (const auto& path : paths) {
			//Get names of files
			QStringList allFiles;
			QDir dir(path);
			dir.setFilter(QDir::Files);
			QFileInfoList list = dir.entryInfoList();
			for (int i = 0; i < list.size(); ++i) {
				QFileInfo dirInfo = list.at(i);
				QString fileName = dirInfo.fileName();
				if (!fileName.endsWith(".tif")) continue;
				std::cout << fileName.toStdString() << std::endl;

				/*poca::core::BasicComponentInterface* bci = engine->loadData(path + fileName);
				if (bci == NULL) {
					std::cout << "Loading image failed" << std::endl;
					continue;
				}
				poca::core::ImagesList* imlist = static_cast <poca::core::ImagesList*>(bci);
				*/

				std::cout << (path + fileName).toStdString() << std::endl;

				poca::core::CommandInfo ci;
				poca::core::MyObjectInterface* obj = engine->loadDataAndCreateObject(path + fileName, &ci);
				if (obj == NULL || !obj->hasBasicComponent("ImagesList")) {
					std::cout << "Loading actin image failed" << std::endl;
					continue;
				}
				poca::core::ImagesList* imlist = static_cast <poca::core::ImagesList*>(obj->getBasicComponent("ImagesList"));

				plugins->addCommands(imlist);

				poca::core::Image<uint16_t>* labels = static_cast<poca::core::Image<uint16_t>*>(imlist->currentImage());
				if (labels == NULL) {
					std::cout << "labels should be uint16_t" << std::endl;
					continue;
				}
				if (labels->depth() == 1) {
					std::cout << "problem with this image" << std::endl;
					continue;
				}

				plugins->addCommands(labels);
				engine->executeCommand(imlist, false, "thresholdImage", "thresholdMin", 1.5f, "thresholdMax", std::numeric_limits<uint16_t>::max());
				engine->executeCommand(imlist->currentImage(), false, "scaleZ", 3.f);
				engine->executeCommand(imlist, false, "marchingCubes", "threshold", 0.5f, "repair", true, "remeshing", true, "targetLength", 4.f, "iterations", (uint32_t)3, "inROIs", false, "scaleZ", 3.f);

				if(!obj->hasBasicComponent("ObjectLists")){
					std::cout << "problem with marching cube" << std::endl;
					delete obj;
					continue;
				}
				poca::geometry::ObjectLists* objlist = static_cast <poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
				poca::geometry::ObjectListMesh* omeshes = static_cast <poca::geometry::ObjectListMesh*>(objlist->currentObjectList());

				meshes.push_back(omeshes->getMeshes().front());

				// Smooth with both angle and area criteria + Delaunay flips
				//CGAL::Polygon_mesh_processing::angle_and_area_smoothing(meshes.back(), CGAL::parameters::number_of_iterations(3).use_safety_constraints(false));
				poca::geometry::laplacian_smooth(meshes.back(), 5, 0.5);

				auto bbox = CGAL::bounding_box(meshes.back().points().begin(), meshes.back().points().end());
				bboxes.emplace_back(bbox.xmin(), bbox.ymin(), bbox.zmin(), bbox.xmax() + PADDING_BINS, bbox.ymax() + PADDING_BINS, bbox.zmax());
				rects.push_back({ cur++, (int)(bbox.xmax() - bbox.xmin() + PADDING_BINS), (int)(bbox.ymax() - bbox.ymin() + PADDING_BINS), 0, 0, 0 });
			}
		}

		Bin bin{ 512, 512 };            // start size
		const int MAX_W = 8192;       // safety caps (tune as needed)
		const int MAX_H = 8192;

		// Retry with growth until success or we hit caps.
		while (true) {
			// Make a working copy because stbrp_rect gets filled in-place (x,y,was_packed).
			auto work = rects;
			if (try_pack(bin, work)) {
				//std::cout << "Packed in " << bin.w << "x" << bin.h << "\n";
				for (const auto& r : work) {
					// was_packed should be 1 for all if try_pack succeeded
					//std::cout << "id=" << r.id << " x=" << r.x << " y=" << r.y << " w=" << r.w << " h=" << r.h << "\n";
					float w = bboxes[r.id].realWidth(), h = bboxes[r.id].realHeight();
					bboxes[r.id].set(r.x, r.y, 0.f, r.x + w, r.y + h, 0.f);
					/*std::cout << r.id << " -> " << m_spinesTriangles[r.id].size();
					for (const auto& tmp : m_spinesTriangles[r.id])
						std::cout << " ; " << tmp->info().m_index;
					std::cout << std::endl;*/
				}
				break;
			}

			Bin next = grow(bin, MAX_W, MAX_H);
			if (next.w == bin.w && next.h == bin.h) {
				std::cerr << "Cannot fit within caps " << MAX_W << "x" << MAX_H << "\n";
				return;
			}
			bin = next;
		}

		for (auto n = 0; n < meshes.size(); n++) {
			Kernel::Vector_3 t(bboxes[n][0], bboxes[n][1], 0.f);
			for (auto v : meshes[n].vertices()) {
				meshes[n].point(v) = meshes[n].point(v) + t;
			}
		}

		poca::geometry::ObjectListMesh* finalOmesh = new poca::geometry::ObjectListMesh(meshes);
		poca::core::CommandInfo com;
		poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(finalOmesh, com, "MainWindow");
		poca::core::MyObjectInterface* obj = engine->createObject("D:/DataPALMSTORM/nucleus_segmentation/all_masks_singapour/", "all_masks.obj", objsList);
		if (obj == NULL)
			return;
		poca::opengl::CameraInterface* cam = createWindows(obj);
		engine->addCameraToObject(obj, cam);

	}
	else if (tmp == "embedShapeLatentSpace") {
		QString pathPolygons("M:/everyone/embedShape");
		QStringList allFiles;
		QDir dir(pathPolygons);
		dir.setFilter(QDir::Files);
		QFileInfoList list = dir.entryInfoList();
		std::vector <poca::core::MyObjectInterface*> objects;
		for (int i = 0; i < list.size(); ++i) {
			QFileInfo dirInfo = list.at(i);
			QString fileName = dirInfo.filePath();
			if (!fileName.endsWith(".csv")) continue;
			//std::cout << fileName.toStdString() << std::endl;
			std::ifstream fs(fileName.toLatin1().data());
			std::string s;
			std::vector <Point_2> points, translatedPoints;
			float xmin = FLT_MAX, xmax = -FLT_MAX, ymin = FLT_MAX, ymax = -FLT_MAX;
			while (std::getline(fs, s)) {
				std::vector < std::string > values;
				values = poca::core::split(s, ',', values);
				float x = ::atof(values[0].c_str()), y = ::atof(values[1].c_str());
				points.emplace_back(x, y);
				if (x < xmin) xmin = x;
				if (x > xmax) xmax = x;
				if (y < ymin) ymin = y;
				if (y > ymax) ymax = y;
			}
			fs.close();
			for (auto& pt : points) 
				translatedPoints.emplace_back(pt.x() - xmin, pt.y() - ymin);
			Polygon_2 poly(translatedPoints.begin(), translatedPoints.end());
			poca::geometry::ObjectListPolygon* objPoly = new poca::geometry::ObjectListPolygon(std::vector <Polygon_2> { poly });
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(objPoly, poca::core::CommandInfo(), "MainWindow::embedShapeLatentSpace");
			auto* polyObj = engine->createObject(dirInfo.absolutePath().toStdString(), dirInfo.completeBaseName().toStdString(), objsList);
			if (polyObj != NULL)
				objects.push_back(polyObj);
			if (i % 250 == 0) {
				std::cout << "Polygon " << i << std::endl;
			}
		}
		QString latentspacePath("M:/everyone/embedShapeResults/results_all_the_data/dataframe_modified.csv");
		std::vector <std::string> names;
		std::vector <poca::core::Vec3mf> translations;
		std::ifstream fs(latentspacePath.toLatin1().data());
		std::string s;
		std::getline(fs, s);
		while (std::getline(fs, s)) {
			std::vector < std::string > values;
			values = poca::core::split(s, ',', values);
			names.push_back(values[2]);
			translations.emplace_back(::atof(values[values.size() - 2].c_str()) * 1000.f, ::atof(values[values.size() - 1].c_str()) * 1000.f, 0.f);//UMAP
			//translations.emplace_back(::atof(values[3].c_str()) * 100.f, ::atof(values[4].c_str()) * 100.f, 0.f);//PCA
		}
		fs.close();

		if (translations.size() != objects.size())
			std::cout << "This is a big problem" << std::endl;

		std::vector<poca::geometry::PackingCircle> circles;
		for (auto n = 0; n < objects.size(); n++) {
			auto obj = objects.at(n);
			const auto& t = translations.at(n);
			const auto& bbox = obj->boundingBox();
			const auto& centroid = bbox.centroid();
			float r = (poca::core::Vec3mf(bbox[3], bbox[4], bbox[5]) - poca::core::Vec3mf(bbox[0], bbox[1], bbox[2])).length() / 2.f;

			circles.push_back({ t[0] + centroid[0], t[1] + centroid[1], r, n });
		}

		int iterations = poca::geometry::relaxationCircles(circles, 200, 0.0015f, 0.55f);
		std::cout << "# iterations = " << iterations << std::endl;

		for (auto n = 0; n < objects.size(); n++) {
			auto obj = objects.at(n);
			auto& t = translations.at(n);
			const auto& bbox = obj->boundingBox();
			const auto& centroid = bbox.centroid();
			const auto& c = circles.at(n);

			t.set(c.x - centroid[0], c.y - centroid[1], t[0]);
		}

		MyMultipleObject* mobjects = static_cast <MyMultipleObject*>(engine->generateMultipleObject(objects));
		std::vector <poca::core::BoundingBox>& gbboxes = mobjects->getGridBBoxes();
		for (auto n = 0; n < mobjects->nbColors(); n++) {
			auto* curObj = mobjects->getObject(n);
			const auto& bbox = curObj->boundingBox();
			auto w = bbox.realWidth(), h = bbox.realHeight(), d = bbox.realThick();
			gbboxes[n].set(translations[n][0], translations[n][1], translations[n][2], translations[n][0] + w, translations[n][1] + h, translations[n][2]);
		}
		mobjects->resetModelMatrices(true);

		poca::opengl::CameraInterface* cam = createWindows(mobjects);
		engine->addCameraToObject(mobjects, cam);
		cam->makeCurrent();
		poca::core::CommandInfo ci(false, "createDisplay");
		mobjects->executeGlobalCommand(&ci);
	}
	else if (tmp == "embedShapeLatentSpaceExport") {
		auto obj = m_currentMdi->getWidget()->getObject();
		auto mobjects = dynamic_cast <MyMultipleObject*>(obj);
		if (!mobjects) return;
		std::vector <poca::core::BoundingBox>& gbboxes = mobjects->getGridBBoxes();
		std::vector <Polygon_2> polygons;
		for (auto n = 0; n < mobjects->nbColors(); n++) {
			auto o = mobjects->getObject(n);
			if (!o->hasBasicComponent("ObjectLists")) continue;
			auto olist = static_cast <poca::geometry::ObjectLists*>(o->getBasicComponent("ObjectLists"));
			if (olist->nbComponents() == 0) continue;
			poca::geometry::ObjectListPolygon* opoly = dynamic_cast <poca::geometry::ObjectListPolygon*>(olist->getComponent(0));
			if (!opoly) continue;
			const auto& vpol = opoly->getPolygons();
			const auto& t = gbboxes[n];
			for (const auto& pol : vpol) {
				auto newpoly = pol.at(0);
				CGAL::Aff_transformation_2<K_inexact> translate(CGAL::TRANSLATION, CGAL::Vector_2<K_inexact> (t.x(), t.y()));
				newpoly = CGAL::transform(translate, newpoly);
				polygons.push_back(newpoly);
			}
		}
		poca::geometry::ObjectListPolygon* objPoly = new poca::geometry::ObjectListPolygon(polygons);
		poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(objPoly, poca::core::CommandInfo(), "MainWindow::embedShapeLatentSpaceExport");
		auto* polyObj = engine->createObject(mobjects->getDir(), "embedAllObjects", objsList);
		poca::opengl::CameraInterface* cam = createWindows(polyObj);
		engine->addCameraToObject(polyObj, cam);
		cam->makeCurrent();
		poca::core::CommandInfo ci(false, "createDisplay");
		polyObj->executeGlobalCommand(&ci);
	}
	else if (tmp == "locsDomino") {
		auto obj = m_currentMdi->getWidget()->getObject();
		auto mobjects = dynamic_cast <MyMultipleObject*>(obj);
		if (!mobjects) return;
		std::vector <float> xs, ys, zs, frame, label;
		for (auto n = 0; n < mobjects->nbColors(); n++) {
			auto obj = mobjects->getObject(n);
			if (!obj->hasBasicComponent("DetectionSet")) continue;
			poca::geometry::DetectionSet* dset = static_cast <poca::geometry::DetectionSet*>(obj->getBasicComponent("DetectionSet"));
			std::vector <float>& oxs = dset->getMyData("x")->getData<float>();
			std::vector <float>& oys = dset->getMyData("y")->getData<float>();
			std::vector <float>& ozs = dset->getMyData("z")->getData<float>();
			std::vector <float>& oframes = dset->getMyData("frame")->getData<float>();
			std::vector <float> olabel(oxs.size(), n);
			
			std::copy(oxs.begin(), oxs.end(), std::back_inserter(xs));
			std::copy(oys.begin(), oys.end(), std::back_inserter(ys));
			std::copy(ozs.begin(), ozs.end(), std::back_inserter(zs));
			std::copy(oframes.begin(), oframes.end(), std::back_inserter(frame));
			std::copy(olabel.begin(), olabel.end(), std::back_inserter(label));
		}
	
		std::map <std::string, std::vector <float>> features;

		features["x"] = xs;
		features["y"] = ys;
		features["z"] = zs;
		features["frame"] = frame;
		features["label"] = label;

		poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(features);

		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::PluginList* plugins = engine->getPlugins();
		poca::core::MyObjectInterface* newobj = engine->createObject(mobjects->getDir(), "merged.csv", dset);
		if (newobj == NULL)
			return;
		poca::opengl::CameraInterface* cam = createWindows(newobj);
		engine->addCameraToObject(newobj, cam);
	}
	else if (tmp == "objsDomino") {
		auto obj = m_currentMdi->getWidget()->getObject();
		auto mobjects = dynamic_cast <MyMultipleObject*>(obj);
		if (!mobjects) return;
		std::vector <Surface_mesh_3_double> meshes;
		std::vector <float> label;
		for (auto n = 0; n < mobjects->nbColors(); n++) {
			auto obj = mobjects->getObject(n);
			if (!obj->hasBasicComponent("ObjectLists")) continue;
			poca::geometry::ObjectLists* olist = static_cast <poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			poca::geometry::ObjectListMesh* omeshes = dynamic_cast<poca::geometry::ObjectListMesh*>(olist->currentObjectList());
			if (omeshes == NULL) continue;
			const auto& orig_meshes = omeshes->getMeshes();
			std::vector <float> olabel(orig_meshes.size(), n);

			std::copy(orig_meshes.begin(), orig_meshes.end(), std::back_inserter(meshes));
			std::copy(olabel.begin(), olabel.end(), std::back_inserter(label));
		}

		poca::geometry::ObjectListMesh* finalMeshes = new poca::geometry::ObjectListMesh(meshes);
		finalMeshes->addFeature("label", poca::core::generateDataWithLog(label));
		poca::geometry::ObjectLists* newObjsList = new poca::geometry::ObjectLists(finalMeshes, poca::core::CommandInfo(), "MainWindow");

		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::PluginList* plugins = engine->getPlugins();
		poca::core::MyObjectInterface* newobj = engine->createObject(mobjects->getDir(), "merged.obj", newObjsList);
		if (newobj == NULL)
			return;
		poca::opengl::CameraInterface* cam = createWindows(newobj);
		engine->addCameraToObject(newobj, cam);
		}
	else if (tmp == "domino_exportColorsByCouple") {
		auto obj = m_currentMdi->getWidget()->getObject();
		if (!obj->hasBasicComponent("DetectionSet")) return;
		poca::geometry::DetectionSet* dset = static_cast <poca::geometry::DetectionSet*>(obj->getBasicComponent("DetectionSet"));
		std::vector <std::string> labels = { "gamma2", "gephyrin", "gluN1", "munc13", "psd95" };
		std::vector <std::array <int, 2>> couples = { {2, 0}, {2, 1}, {2, 3}, {2, 4} };
		std::vector <float>& oxs = dset->getMyData("x")->getData<float>();
		std::vector <float>& oys = dset->getMyData("y")->getData<float>();
		std::vector <float>& ozs = dset->getMyData("z")->getData<float>();
		std::vector <float>& oframes = dset->getMyData("frame")->getData<float>();
		std::vector <float> olabel = dset->getMyData("label")->getData<float>();

		const std::string dir = obj->getDir();
		const std::string& name = obj->getName();

		for (const auto& couple : couples) {
			std::vector <float> xs, ys, zs, frame, label;
			for (auto n = 0; n < oxs.size(); n++) {
				if (olabel[n] == couple[0] || olabel[n] == couple[1]) {
					xs.push_back(oxs[n]);
					ys.push_back(oys[n]);
					zs.push_back(ozs[n]);
					frame.push_back(oframes[n]);
					if (olabel[n] == couple[0])
						label.push_back(1);
					else
						label.push_back(2);
				}
			}
			std::map <std::string, std::vector <float>> features;

			features["x"] = xs;
			features["y"] = ys;
			features["z"] = zs;
			features["frame"] = frame;
			features["label"] = label;

			poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(features);

			std::string filename = dir + "/" + name + "_" + labels[couple[0]] + "_" + labels[couple[1]] + ".csv";
			dset->saveDetections(filename);
			delete dset;
		}
	}
}

void MainWindow::onGridReleased()
{
	if (m_currentMdi == NULL) return;
	MyMultipleObject* mobject = dynamic_cast <MyMultipleObject*>(m_currentMdi->getWidget()->getObject());
	if (!mobject) return;

	mobject->recomputeGrid();
	mobject->notifyAll("updateDisplay");
	resetViewer();
}

void MainWindow::onToggleGridCentered(bool _on)
{
	if (m_currentMdi == NULL) return;
	MyMultipleObject* mobject = dynamic_cast <MyMultipleObject*>(m_currentMdi->getWidget()->getObject());
	if (!mobject) return;

	mobject->resetModelMatrices(_on);
	mobject->executeGlobalCommand(&poca::core::CommandInfo(false, "rebuildDisplay"));
	mobject->notifyAll("updateDisplay");
	resetViewer();
}

void MainWindow::onExportAllObjects()
{
	if (m_currentMdi == NULL) return;
	MyMultipleObject* mobject = dynamic_cast <MyMultipleObject*>(m_currentMdi->getWidget()->getObject());
	if (!mobject) return;

	std::vector <Surface_mesh_3_double> allObjects;
	for (auto n = 0; n < mobject->nbColors(); n++) {
		auto curobj = mobject->getObject(n);
		if (!curobj->hasBasicComponent("ObjectLists")) continue;
		auto bci = curobj->getBasicComponent("ObjectLists");
		poca::core::BasicComponentList* bclist = static_cast <poca::core::BasicComponentList*>(bci);
		poca::geometry::ObjectListMesh* omesh = dynamic_cast <poca::geometry::ObjectListMesh*>(bclist->currentComponent());
		const auto& meshes = omesh->getMeshes();
		for (const auto& mesh : meshes) {
			allObjects.push_back(mesh);
			auto& addedMesh = allObjects.back();
			//poca::geometry::laplacian_smooth(addedMesh, 5, 0.5);
			auto bbox = CGAL::Polygon_mesh_processing::bbox(addedMesh);
			std::cout << bbox.xmin() << " " << bbox.ymin() << " " << bbox.zmin() << " " << bbox.xmax() << " " << bbox.ymax() << " " << bbox.zmax() << std::endl;
			Kernel::Vector_3 t(bbox.xmin(), bbox.ymin(), bbox.zmin());
			for (auto v : addedMesh.vertices()) {
				addedMesh.point(v) = addedMesh.point(v) - t;
			}
			bbox = CGAL::Polygon_mesh_processing::bbox(addedMesh);
			std::cout << bbox.xmin() << " " << bbox.ymin() << " " << bbox.zmin() << " " << bbox.xmax() << " " << bbox.ymax() << " " << bbox.zmax() << std::endl;
		}
	}

	std::vector <poca::core::BoundingBox> bboxes;
	std::vector<stbrp_rect> rects;
	int cur = 0;
	uint32_t PADDING_BINS = 0;

	for (auto& mesh : allObjects) {
		auto bbox = CGAL::Polygon_mesh_processing::bbox(mesh);
		std::cout << bbox.xmin() << " " << bbox.ymin() << " " << bbox.zmin() << " " << bbox.xmax() << " " << bbox.ymax() << " " << bbox.zmax() << std::endl;
		bboxes.emplace_back(bbox.xmin(), bbox.ymin(), bbox.zmin(), bbox.xmax() + PADDING_BINS, bbox.ymax() + PADDING_BINS, bbox.zmax());
		rects.push_back({ cur++, (int)std::ceil(bbox.xmax() - bbox.xmin() + PADDING_BINS), (int)std::ceil(bbox.ymax() - bbox.ymin() + PADDING_BINS), 0, 0, 0 });
	}

	Bin bin{ 512, 512 };            // start size
	const int MAX_W = 50000;       // safety caps (tune as needed)
	const int MAX_H = 50000;

	// Retry with growth until success or we hit caps.
	while (true) {
		// Make a working copy because stbrp_rect gets filled in-place (x,y,was_packed).
		auto work = rects;
		if (try_pack(bin, work)) {
			//std::cout << "Packed in " << bin.w << "x" << bin.h << "\n";
			for (const auto& r : work) {
				float w = bboxes[r.id].realWidth(), h = bboxes[r.id].realHeight();
				bboxes[r.id].set(r.x, r.y, 0.f, r.x + w, r.y + h, 0.f);
			}
			break;
		}

		Bin next = grow(bin, MAX_W, MAX_H);
		if (next.w == bin.w && next.h == bin.h) {
			std::cerr << "Cannot fit within caps " << MAX_W << "x" << MAX_H << "\n";
			return;
		}
		bin = next;
	}

	for (auto n = 0; n < allObjects.size(); n++) {
		Kernel::Vector_3 t(bboxes[n][0], bboxes[n][1], 0.f);
		for (auto v : allObjects[n].vertices()) {
			allObjects[n].point(v) = allObjects[n].point(v) + t;
		}
	}

	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::geometry::ObjectListMesh* finalOmesh = new poca::geometry::ObjectListMesh(allObjects);
	poca::core::CommandInfo com;
	poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(finalOmesh, com, "MainWindow");
	poca::core::MyObjectInterface* obj = engine->createObject(mobject->getDir(), "all_objects.obj", objsList);
	if (obj == NULL)
		return;
	poca::opengl::CameraInterface* cam = createWindows(obj);
	engine->addCameraToObject(obj, cam);
}

void MainWindow::reloadCurrentDataset()
{
	if (m_currentMdi == NULL) return;
	m_currentMdi->getWidget()->getObject()->notify("LoadObjCharacteristicsAllWidgets");
}
