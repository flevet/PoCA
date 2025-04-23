/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MainWindow.hpp
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

#ifndef MainWindow_h__
#define MainWindow_h__

#include <Windows.h>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QLabel>
#include <glm/glm.hpp>

#include <DesignPatterns/Observer.hpp>
#include <General/json.hpp>
#include  <General/Vec6.hpp>

class QMdiArea;
class QProgressBar;
class QPlainTextEdit;
class QMdiSubWindow;
class QSignalMapper;
class QPushButton;
class QButtonGroup;
class QAbstractButton;
class QImage;

class MdiChild;
class MainFilterWidget;
class PythonWidget;
class ROIGeneralWidget;
class MacroWidget;

class LoaderLocalizationsInterface;
class LoaderImageInterface;
class GuiInterface;
class PluginInterface;

namespace poca::core {
	class PluginList;
	class BasicComponent;
}

namespace poca::geometry {
	class DetectionSet;
}

namespace poca::opengl {
	class CameraInterface;
}

class MainWindow : public QMainWindow, public poca::core::Observer {
	Q_OBJECT

public:
	MainWindow();
	~MainWindow();

	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&);

	inline void setPathToFileToOpen(const QString & _path){ m_pathToFileToOpen = _path; }
	inline const QString & getPathToFileToOpen() const{ return m_pathToFileToOpen; }

protected:
	void closeEvent(QCloseEvent *);
	void keyPressEvent(QKeyEvent *);
	void dragEnterEvent(QDragEnterEvent*);
	void dropEvent(QDropEvent*);

	MdiChild * getChild(const unsigned int);
	QWidget * getFilterWidget(const QString &);

	void computeColocalization(const int, const int);
	void computeColocalization(const std::vector < std::string>&);
	void computeColocalization(const std::vector < MdiChild*>&);

	void openFile(const QString &, poca::core::CommandInfo*);

	poca::opengl::CameraInterface* createWindows(poca::core::MyObjectInterface*);
	poca::core::MyObjectInterface* createWindows(poca::core::BasicComponent*, const QString&, const QString&);
	void updateTabWidget();

	void execute(poca::core::CommandInfo*);
	void runMacro(const nlohmann::json&);

	void savePositionCamera();
	void savePositionCamera(const std::string&);
	void loadPositionCamera();
	void loadPositionCamera(const std::string&, const bool = false, const bool = true, const bool = true, const bool = true, const bool = true, const bool = true);

private slots:
	void actionNeeded();
	void openFile();
	void openDir();
	void addComponentToCurrentMdi();
	void setActiveMdiChild( MdiChild * );
	void toggleGridDisplay();
	void toggleFontDisplay();
	void aboutDialog();
	void closeAllDatasets();
	void tileWindows();
	void cascadeWindows();
	void actionFromPlugin();
	void resetViewer();
	void toggleBoundingBoxDisplay();
	void computeColocalization();
	void changeColorObject(QAbstractButton*);
	void duplicate();
	void setCameraInteraction();
	void setCameraInteraction(bool);
	void createWidget(poca::core::MyObjectInterface*);

	
	void savePositionCameraSlot(QString);
	void loadPositionCameraSlot(QString);
	void pathCameraSlot(QString, QString, float, bool, bool);
	void pathCameraSlot2(nlohmann::json, nlohmann::json, float, bool, bool);
	void pathCameraAllSlot(const std::vector <std::tuple<float, glm::vec3, glm::quat>>&, bool, bool);

	void setParametersPython();

	void currentCameraForPath();


public slots:
	void setPermanentStatusText(const QString &);
	void runMacro(std::vector<nlohmann::json>);
	void runMacro(std::vector<nlohmann::json>, QStringList);
	void createObjectFromFeatures(const std::map <std::string, std::vector <float>>&, const std::string, const std::string);
	void createMovie();
	void zoomToCropCurrentMdi(poca::core::BoundingBox);

private:
	void createActions();
	void createToolBars();
	void createStatusBar();
	void createMenus();

	QMdiArea * m_mdiArea;
	QSignalMapper * m_windowMapper;

	QToolBar * m_fileToolBar;
	QAction * m_openFileAct, * m_openDirAct, * m_plusAct, * m_duplicateAct, * m_exitAct, * m_gridAct, * m_fontDisplayAct, * m_colocAct, * m_aboutAct, * m_resetProjAct;
	QAction * m_closeAllAct, * m_boundingBoxAct;
	QAction * m_tileWindowsAct, *m_cascadeWindowsAct;
	QAction* m_line2DROIAct, * m_triangle2DROIAct, * m_circle2DROIAct, * m_square2DROIAct, * m_polyline2DROIAct, * m_sphere3DROIAct, * m_planeROIAct, * m_polyplaneROIAct;
	QAction* m_cropAct, * m_xyAct, * m_xzAct, * m_yzAct;
	QAction* m_pythonParamsAct;
	QTabWidget * m_tabWidget;
	QProgressBar * m_progressBar;
	QLabel * m_lblPermanentStatus;
	QAction* m_lastActionQuantifToolbar, *m_lastActionColocToolbar, *m_lastActionDisplayToolbar, *m_lastActionMiscToolbar, * m_lastActionROIToolbar;

	bool m_firstLoad;
	QString m_pathToFileToOpen;

	MainFilterWidget * m_mfw;
	PythonWidget* m_pythonW; 
	ROIGeneralWidget* m_ROIsW;
	MacroWidget* m_macroW;
	MdiChild * m_currentMdi;

	double m_infosCreationDatasets[8];

	QWidget* m_widgetColors;
	std::vector <QPushButton*> m_colorButtons;
	QButtonGroup* m_colorButtonsGroup;

	uint32_t m_currentDuplicate;
};

#endif // MainWindow_h__ 

