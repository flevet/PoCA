/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSetWidget.hpp
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

#ifndef DetectionSetWidget_h__
#define DetectionSetWidget_h__

#include <QtWidgets/QTabWidget>

#include <Plot/FilterHistogramWidget.hpp>
#include <DesignPatterns/Observer.hpp>
#include <General/Command.hpp>
#include <General/Palette.hpp>
#include <Interfaces/MyObjectInterface.hpp>

class QPushButton;
class QCheckBox;
class QPlainTextEdit;

namespace poca::core {
	class EquationFit;
}

//! [0]
class DetectionSetWidget : public QWidget, public poca::core::ObserverForMediator {
	Q_OBJECT

public:
	DetectionSetWidget(poca::core::MediatorWObjectFWidgetInterface*, QWidget* = 0);
	~DetectionSetWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&);
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

	QWidget* getCleanerWidget() { return m_cleanerWidget; }

protected:
	void fillPlot(QCustomPlot*, poca::core::EquationFit*);
	inline int pointSize() const { return m_sizePointSpn->value(); }

signals:
	void transferNewObjectCreated(poca::core::MyObjectInterface*);

protected slots:
	void actionNeeded();
	void actionNeeded(bool);
	void actionNeeded(int);

protected:
	QTabWidget* m_parentTab;
	poca::core::MediatorWObjectFWidgetInterface* m_mediator;

	//Filtering/display
	QWidget* m_lutsWidget, * m_detectionSetFilteringWidget, *m_emptyWidget, * m_line2Widget;
	std::vector <std::pair<QPushButton*, std::string>> m_lutButtons, m_lutHeatmapButtons;
	std::vector <poca::plot::FilterHistogramWidget*> m_histWidgets;
	QPushButton * m_displayButton, * m_heatmapButton, * m_pointRenderButton, * m_saveDetectionsButton, * m_gaussianButton, * m_parametersButton, * m_creationObjectsOnLabelsButton;
	QPushButton* m_worldButton, * m_screenButton;
	QLineEdit* m_minRadiusEdit, * m_maxRadiusEdit, * m_currentRadiusEdit, * m_intensityEdit;
	QSlider* m_radiusSlider, * m_intensitySlider;
	QCheckBox* m_radiusScreenHeatCbox, * m_radiusWorldHeatCbox, * m_interpolateLUTHeatmapCbox;
	QLabel* m_nbLocsLbl;
	QButtonGroup* m_buttonGroup, * m_worldScreenbuttonGroup;

	QSpinBox* m_sizePointSpn;

	QGroupBox* m_groupBoxHeatmap, * m_groupBoxGaussian;

	QSlider* m_alphaGaussianSlider;
	QLabel* m_alphaValueLbl;
	QCheckBox* m_fixedSizeGaussCBox;

	//Cleaner
	QLineEdit* m_radiusCleanerEdit, * m_maxDarkTEdit;
	QCheckBox* m_fixedDarkTcbox;
	QPushButton* m_cleanButton, * m_displayCleanButton, * m_saveFramesButton;
	QCustomPlot* m_plotBlinks, * m_plotTOns, * m_plotToffs;
	QGroupBox* m_groupBoxCleanerPlots;
	QPlainTextEdit* m_statsTEdit;

	QWidget* m_cleanerWidget;
	poca::core::MyObjectInterface* m_object;
};

//! [0]
#endif

