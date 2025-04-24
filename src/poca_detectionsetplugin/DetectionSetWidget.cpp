/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSetWidget.cpp
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

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QButtonGroup>
#include <iostream>
#include <fstream>

#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/CommandableObjectInterface.hpp>
#include <General/BasicComponent.hpp>
#include <Plot/Icons.hpp>
#include <Plot/Misc.h>
#include <General/EquationFit.hpp>
#include <Geometry/DetectionSet.hpp>
#include <General/CommandableObject.hpp>

#include "DetectionSetWidget.hpp"

DetectionSetWidget::DetectionSetWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, QWidget* _parent):m_object(NULL)// :QTabWidget(_parent)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;

	this->setObjectName("DetectionSetWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsDetectionSetWidget");
	this->addActionToObserve("UpdateHistogramDetectionSetWidget");
	int maxSize = 20;

	m_lutsWidget = new QWidget;
	m_lutsWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	QHBoxLayout* layoutLuts = new QHBoxLayout, * layoutLine2 = new QHBoxLayout;
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("HotCold2")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("InvFire")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("Fire")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("Ice")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllRedColorBlind")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllGreenColorBlind")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllRed")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllBlue")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllWhite")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllBlack")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("Gray")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("Random")));
	for (size_t n = 0; n < m_lutButtons.size(); n++) {
		m_lutButtons[n].first->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_lutButtons[n].first->setMaximumSize(QSize(maxSize, maxSize));
		QImage im = poca::core::generateImage(maxSize, maxSize, &poca::core::Palette::getStaticLut(m_lutButtons[n].second));
		QPixmap pix = QPixmap::fromImage(im);
		QIcon icon(pix);
		m_lutButtons[n].first->setIcon(icon);
		if (n < 9)
			layoutLuts->addWidget(m_lutButtons[n].first, 0, Qt::AlignLeft);
		else
			layoutLine2->addWidget(m_lutButtons[n].first);
		QObject::connect(m_lutButtons[n].first, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	}
	QWidget* emptyLuts = new QWidget;
	emptyLuts->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	layoutLuts->addWidget(emptyLuts);

	m_saveDetectionsButton = new QPushButton();
	m_saveDetectionsButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_saveDetectionsButton->setMaximumSize(QSize(maxSize, maxSize));
	m_saveDetectionsButton->setIcon(QIcon(QPixmap(poca::plot::saveIcon)));
	m_saveDetectionsButton->setToolTip("Save detections");
	layoutLuts->addWidget(m_saveDetectionsButton, 0, Qt::AlignRight);
	QObject::connect(m_saveDetectionsButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	m_pointRenderButton = new QPushButton();
	m_pointRenderButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_pointRenderButton->setMaximumSize(QSize(maxSize, maxSize));
	m_pointRenderButton->setIcon(QIcon(QPixmap(poca::plot::pointRenderingIcon)));
	m_pointRenderButton->setToolTip("Render points");
	m_pointRenderButton->setCheckable(true);
	m_pointRenderButton->setChecked(true);
	layoutLuts->addWidget(m_pointRenderButton, 0, Qt::AlignRight);
	QObject::connect(m_pointRenderButton, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	
	m_heatmapButton = new QPushButton();
	m_heatmapButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_heatmapButton->setMaximumSize(QSize(maxSize, maxSize));
	m_heatmapButton->setIcon(QIcon(QPixmap(poca::plot::heatmapIcon)));
	m_heatmapButton->setToolTip("Toggle heatmap");
	m_heatmapButton->setCheckable(true);
	m_heatmapButton->setChecked(true);
	layoutLuts->addWidget(m_heatmapButton, 0, Qt::AlignRight);
	QObject::connect(m_heatmapButton, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));

	m_gaussianButton = new QPushButton();
	m_gaussianButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_gaussianButton->setMaximumSize(QSize(maxSize, maxSize));
	m_gaussianButton->setIcon(QIcon(QPixmap(poca::plot::gauss3DIcon)));
	m_gaussianButton->setToolTip("Toggle gaussian rendering");
	m_gaussianButton->setCheckable(true);
	m_gaussianButton->setChecked(true);
	layoutLuts->addWidget(m_gaussianButton, 0, Qt::AlignRight);
	QObject::connect(m_gaussianButton, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));

	m_buttonGroup = new QButtonGroup(this);
	m_buttonGroup->addButton(m_pointRenderButton);
	m_buttonGroup->addButton(m_heatmapButton);
	m_buttonGroup->addButton(m_gaussianButton);

	m_creationObjectsOnLabelsButton = new QPushButton();
	m_creationObjectsOnLabelsButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_creationObjectsOnLabelsButton->setMaximumSize(QSize(maxSize, maxSize));
	m_creationObjectsOnLabelsButton->setIcon(QIcon(QPixmap(poca::plot::objectIcon)));
	m_creationObjectsOnLabelsButton->setToolTip("Create objects");
	layoutLuts->addWidget(m_creationObjectsOnLabelsButton, 0, Qt::AlignRight);
	QObject::connect(m_creationObjectsOnLabelsButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	m_displayButton = new QPushButton();
	m_displayButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_displayButton->setMaximumSize(QSize(maxSize, maxSize));
	m_displayButton->setIcon(QIcon(QPixmap(poca::plot::brushIcon)));
	m_displayButton->setToolTip("Toggle display");
	m_displayButton->setCheckable(true);
	m_displayButton->setChecked(true);
	layoutLuts->addWidget(m_displayButton, 0, Qt::AlignRight);
	QObject::connect(m_displayButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_lutsWidget->setLayout(layoutLuts);

	m_line2Widget = new QWidget;
	QWidget* emptyLine2 = new QWidget;
	emptyLine2->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	layoutLine2->addWidget(emptyLine2);
	QLabel* sizePointLbl = new QLabel();
	sizePointLbl->setMaximumSize(QSize(maxSize, maxSize));
	sizePointLbl->setPixmap(QPixmap(poca::plot::pointSizeIcon).scaled(maxSize, maxSize, Qt::KeepAspectRatio));
	layoutLine2->addWidget(sizePointLbl, 0, Qt::AlignRight);
	m_sizePointSpn = new QSpinBox;
	m_sizePointSpn->setRange(1, 100);
	m_sizePointSpn->setValue(1);
	m_sizePointSpn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QObject::connect(m_sizePointSpn, SIGNAL(valueChanged(int)), this, SLOT(actionNeeded(int)));
	layoutLine2->addWidget(m_sizePointSpn, 0, Qt::AlignRight);
	m_parametersButton = new QPushButton();
	m_parametersButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_parametersButton->setMaximumSize(QSize(maxSize, maxSize));
	m_parametersButton->setIcon(QIcon(QPixmap(poca::plot::parametersIcon)));
	m_parametersButton->setToolTip("Parameter dialog");
	layoutLine2->addWidget(m_parametersButton, 0, Qt::AlignRight);
	QObject::connect(m_parametersButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	m_worldButton = new QPushButton();
	m_worldButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_worldButton->setMaximumSize(QSize(maxSize, maxSize));
	m_worldButton->setIcon(QIcon(QPixmap(poca::plot::worldIcon)));
	m_worldButton->setToolTip("World coordinates");
	m_worldButton->setCheckable(true);
	m_worldButton->setChecked(true);
	layoutLine2->addWidget(m_worldButton, 0, Qt::AlignRight);
	QObject::connect(m_worldButton, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	m_screenButton = new QPushButton();
	m_screenButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_screenButton->setMaximumSize(QSize(maxSize, maxSize));
	m_screenButton->setIcon(QIcon(QPixmap(poca::plot::screenIcon)));
	m_screenButton->setToolTip("Screen coordinates");
	m_screenButton->setCheckable(true);
	m_screenButton->setChecked(false);
	layoutLine2->addWidget(m_screenButton, 0, Qt::AlignRight);
	QObject::connect(m_screenButton, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	m_worldScreenbuttonGroup = new QButtonGroup(this);
	m_worldScreenbuttonGroup->addButton(m_worldButton);
	m_worldScreenbuttonGroup->addButton(m_screenButton);
	m_line2Widget->setLayout(layoutLine2);

	m_detectionSetFilteringWidget = new QWidget;
	m_detectionSetFilteringWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_nbLocsLbl = new QLabel("# localizations:");
	m_nbLocsLbl->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

	m_groupBoxHeatmap = new QGroupBox(tr("Heatmap"));
	m_minRadiusEdit = new QLineEdit("1");
	m_minRadiusEdit->setEnabled(false);
	m_minRadiusEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_minRadiusEdit->setMaximumSize(QSize(maxSize * 2, maxSize));
	m_currentRadiusEdit = new QLineEdit("25");
	m_currentRadiusEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_currentRadiusEdit->setMaximumSize(QSize(maxSize * 2, maxSize));
	QObject::connect(m_currentRadiusEdit, SIGNAL(returnPressed()), SLOT(actionNeeded()));
	m_maxRadiusEdit = new QLineEdit("100");
	m_maxRadiusEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_maxRadiusEdit->setMaximumSize(QSize(maxSize * 2, maxSize));
	QObject::connect(m_maxRadiusEdit, SIGNAL(returnPressed()), SLOT(actionNeeded()));
	m_radiusSlider = new QSlider(Qt::Horizontal);
	m_radiusSlider->setMinimum(1);
	m_radiusSlider->setMaximum(100);
	m_radiusSlider->setSliderPosition(25);
	m_radiusSlider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QObject::connect(m_radiusSlider, SIGNAL(valueChanged(int)), SLOT(actionNeeded(int)));
	QLabel* curRadiusLbl = new QLabel("Current radius (pix):");
	curRadiusLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	
	QLabel* curIntensityLbl = new QLabel("Intensity:");
	curIntensityLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_intensityEdit = new QLineEdit("1");
	m_intensityEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_intensityEdit->setMaximumSize(QSize(maxSize * 2, maxSize));
	QObject::connect(m_intensityEdit, SIGNAL(returnPressed()), SLOT(actionNeeded()));
	QLineEdit* intensityMinEdit = new QLineEdit("0.001");
	intensityMinEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	intensityMinEdit->setEnabled(false);
	QLineEdit* intensityMaxEdit = new QLineEdit("1");
	intensityMaxEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	intensityMaxEdit->setEnabled(false);
	m_intensitySlider = new QSlider(Qt::Horizontal);
	m_intensitySlider->setMinimum(1);
	m_intensitySlider->setMaximum(1000);
	m_intensitySlider->setSliderPosition(1000);
	m_intensitySlider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QObject::connect(m_intensitySlider, SIGNAL(valueChanged(int)), SLOT(actionNeeded(int)));
	QHBoxLayout* layoutH3 = new QHBoxLayout;
	layoutH3->addWidget(intensityMinEdit);
	layoutH3->addWidget(m_intensitySlider);
	layoutH3->addWidget(intensityMaxEdit);

	QHBoxLayout* layoutH2 = new QHBoxLayout;
	m_lutHeatmapButtons.push_back(std::make_pair(new QPushButton(), std::string("HotCold2")));
	m_lutHeatmapButtons.push_back(std::make_pair(new QPushButton(), std::string("InvFire")));
	m_lutHeatmapButtons.push_back(std::make_pair(new QPushButton(), std::string("Fire")));
	m_lutHeatmapButtons.push_back(std::make_pair(new QPushButton(), std::string("Ice")));
	m_lutHeatmapButtons.push_back(std::make_pair(new QPushButton(), std::string("Heatmap")));
	for (size_t n = 0; n < m_lutHeatmapButtons.size(); n++) {
		m_lutHeatmapButtons[n].first->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_lutHeatmapButtons[n].first->setMaximumSize(QSize(maxSize, maxSize));
		QImage im = poca::core::generateImage(maxSize, maxSize, &poca::core::Palette::getStaticLut(m_lutHeatmapButtons[n].second));
		QPixmap pix = QPixmap::fromImage(im);
		QIcon icon(pix);
		m_lutHeatmapButtons[n].first->setIcon(icon);
		layoutH2->addWidget(m_lutHeatmapButtons[n].first, 0, Qt::AlignLeft);

		QObject::connect(m_lutHeatmapButtons[n].first, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	}
	QWidget* emptyCurRadiusW = new QWidget;
	emptyCurRadiusW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	QButtonGroup* bgroup = new QButtonGroup;
	m_radiusScreenHeatCbox = new QCheckBox("Radius screen");
	m_radiusScreenHeatCbox->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
	QObject::connect(m_radiusScreenHeatCbox, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	m_radiusWorldHeatCbox = new QCheckBox("Radius world");
	m_radiusWorldHeatCbox->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
	m_radiusWorldHeatCbox->setChecked(true);
	QObject::connect(m_radiusWorldHeatCbox, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	m_interpolateLUTHeatmapCbox = new QCheckBox("Interpolate LUT");
	m_interpolateLUTHeatmapCbox->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
	m_interpolateLUTHeatmapCbox->setChecked(true);
	QObject::connect(m_interpolateLUTHeatmapCbox, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	bgroup->addButton(m_radiusScreenHeatCbox);
	bgroup->addButton(m_radiusWorldHeatCbox);

	QHBoxLayout* layoutH1 = new QHBoxLayout;
	layoutH1->addWidget(m_minRadiusEdit);
	layoutH1->addWidget(m_radiusSlider);
	layoutH1->addWidget(m_maxRadiusEdit);
	
	layoutH2->addWidget(emptyCurRadiusW);
	layoutH2->addWidget(curIntensityLbl, Qt::AlignRight);
	layoutH2->addWidget(m_intensityEdit, Qt::AlignRight);
	layoutH2->addWidget(curRadiusLbl, Qt::AlignRight);
	layoutH2->addWidget(m_currentRadiusEdit, Qt::AlignRight);

	QHBoxLayout* layoutH4 = new QHBoxLayout;
	layoutH4->addWidget(m_radiusScreenHeatCbox);
	layoutH4->addWidget(m_radiusWorldHeatCbox);
	layoutH4->addWidget(m_interpolateLUTHeatmapCbox);

	QVBoxLayout* layoutHeatmap = new QVBoxLayout;
	layoutHeatmap->addLayout(layoutH1);
	layoutHeatmap->addLayout(layoutH3);
	layoutHeatmap->addLayout(layoutH2);
	layoutHeatmap->addLayout(layoutH4);
	m_groupBoxHeatmap->setLayout(layoutHeatmap);

	m_groupBoxGaussian = new QGroupBox(tr("Gaussian"));
	m_alphaValueLbl = new QLabel("Alpha: 0.1");
	m_alphaValueLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	QLabel * minAlphaLbl = new QLabel("0.001");
	minAlphaLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	minAlphaLbl->setMaximumSize(QSize(maxSize * 2, maxSize));
	QLabel* maxAlphaLbl = new QLabel("1");
	maxAlphaLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	maxAlphaLbl->setMaximumSize(QSize(maxSize * 2, maxSize));
	m_alphaGaussianSlider = new QSlider(Qt::Horizontal);
	m_alphaGaussianSlider->setMinimum(1);
	m_alphaGaussianSlider->setMaximum(1000);
	m_alphaGaussianSlider->setSliderPosition(1000);
	m_alphaGaussianSlider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QObject::connect(m_alphaGaussianSlider, SIGNAL(valueChanged(int)), SLOT(actionNeeded(int)));
	m_fixedSizeGaussCBox = new QCheckBox("Fixed size");
	m_fixedSizeGaussCBox->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
	m_fixedSizeGaussCBox->setChecked(true);
	QObject::connect(m_fixedSizeGaussCBox, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	QHBoxLayout* layoutL1Gaussian = new QHBoxLayout;
	layoutL1Gaussian->addWidget(minAlphaLbl);
	layoutL1Gaussian->addWidget(m_alphaGaussianSlider);
	layoutL1Gaussian->addWidget(maxAlphaLbl);
	QHBoxLayout* layoutL2Gaussian = new QHBoxLayout;
	layoutL2Gaussian->addWidget(m_alphaValueLbl);
	layoutL2Gaussian->addWidget(m_fixedSizeGaussCBox, 0, Qt::AlignRight);
	QVBoxLayout* layoutGaussian = new QVBoxLayout;
	layoutGaussian->addLayout(layoutL1Gaussian);
	layoutGaussian->addLayout(layoutL2Gaussian);
	m_groupBoxGaussian->setLayout(layoutGaussian);


	m_emptyWidget = new QWidget;
	m_emptyWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	QGroupBox* groupBox1 = new QGroupBox(tr("Filtering"));
	QVBoxLayout* vboxgb1 = new QVBoxLayout;
	vboxgb1->setContentsMargins(1, 1, 1, 1);
	vboxgb1->setSpacing(1);
	vboxgb1->addWidget(m_lutsWidget);
	vboxgb1->addWidget(m_line2Widget);
	vboxgb1->addWidget(m_detectionSetFilteringWidget);
	vboxgb1->addWidget(m_nbLocsLbl);
	groupBox1->setLayout(vboxgb1); 

	QVBoxLayout* layout = new QVBoxLayout;
	layout->setContentsMargins(1, 1, 1, 1);
	layout->setSpacing(1);
	layout->addWidget(groupBox1);
	layout->addWidget(m_groupBoxHeatmap);
	layout->addWidget(m_groupBoxGaussian);
	layout->addWidget(m_emptyWidget);
	this->setLayout(layout);

	QGroupBox* groupBoxCleanerParams = new QGroupBox(tr("Parameters"));
	QLabel* radiusCleanerLbl = new QLabel("Radius:");
	radiusCleanerLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_radiusCleanerEdit = new QLineEdit("0.3");
	m_radiusCleanerEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_radiusCleanerEdit->setMaximumSize(QSize(maxSize * 2, maxSize));
	QLabel* maxDarkTLbl = new QLabel("Dark time:");
	maxDarkTLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_maxDarkTEdit = new QLineEdit("10");
	m_maxDarkTEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_maxDarkTEdit->setMaximumSize(QSize(maxSize * 2, maxSize));
	m_fixedDarkTcbox = new QCheckBox("Fixed dark time");
	m_fixedDarkTcbox->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_fixedDarkTcbox->setChecked(false);
	m_cleanButton = new QPushButton("Clean");
	m_cleanButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_cleanButton->setMaximumSize(QSize(maxSize * 2, maxSize));
	QObject::connect(m_cleanButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	QWidget* emptyCleanParamW = new QWidget;
	emptyCleanParamW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	m_displayCleanButton = new QPushButton();
	m_displayCleanButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_displayCleanButton->setMaximumSize(QSize(maxSize, maxSize));
	m_displayCleanButton->setIcon(QIcon(QPixmap(poca::plot::brushIcon)));
	m_displayCleanButton->setToolTip("Toggle display");
	m_displayCleanButton->setCheckable(true);
	m_displayCleanButton->setChecked(true);
	QObject::connect(m_displayCleanButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	m_saveFramesButton = new QPushButton();
	m_saveFramesButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_saveFramesButton->setMaximumSize(QSize(maxSize, maxSize));
	m_saveFramesButton->setIcon(QIcon(QPixmap(poca::plot::saveIcon)));
	m_saveFramesButton->setToolTip("Save frames merged locs");
	QObject::connect(m_saveFramesButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	QHBoxLayout* layoutCleanParam = new QHBoxLayout;
	layoutCleanParam->addWidget(radiusCleanerLbl);
	layoutCleanParam->addWidget(m_radiusCleanerEdit);
	layoutCleanParam->addWidget(maxDarkTLbl);
	layoutCleanParam->addWidget(m_maxDarkTEdit);
	layoutCleanParam->addWidget(m_fixedDarkTcbox);
	layoutCleanParam->addWidget(m_cleanButton);
	layoutCleanParam->addWidget(emptyCleanParamW);
	layoutCleanParam->addWidget(m_displayCleanButton, 0, Qt::AlignRight);
	layoutCleanParam->addWidget(m_saveFramesButton, 0, Qt::AlignRight);
	groupBoxCleanerParams->setLayout(layoutCleanParam);
	QWidget* emptyWidgetClean = new QWidget;
	emptyWidgetClean->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	m_groupBoxCleanerPlots = new QGroupBox(tr("Plots"));
	QFont fontLegend("Helvetica", 9);
	fontLegend.setBold(true);
	QColor background = QWidget::palette().color(QWidget::backgroundRole());
	QLabel* blinksPlotLbl = new QLabel("# blinks");
	blinksPlotLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_plotBlinks = new QCustomPlot(this);
	m_plotBlinks->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_plotBlinks->setMinimumHeight(150);
	m_plotBlinks->legend->setTextColor(Qt::black);
	m_plotBlinks->legend->setFont(fontLegend);
	m_plotBlinks->legend->setBrush(Qt::NoBrush);
	m_plotBlinks->legend->setBorderPen(Qt::NoPen);
	m_plotBlinks->setBackground(background);
	QLabel* tonsPlotLbl = new QLabel("# ons");
	tonsPlotLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_plotTOns = new QCustomPlot(this);
	m_plotTOns->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_plotTOns->setMinimumHeight(150);
	m_plotTOns->legend->setTextColor(Qt::black);
	m_plotTOns->legend->setFont(fontLegend);
	m_plotTOns->legend->setBrush(Qt::NoBrush);
	m_plotTOns->legend->setBorderPen(Qt::NoPen);
	m_plotTOns->setBackground(background);
	QLabel* toffsPlotLbl = new QLabel("# offs");
	toffsPlotLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_plotToffs = new QCustomPlot(this);
	m_plotToffs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_plotToffs->setMinimumHeight(150);
	m_plotToffs->legend->setTextColor(Qt::black);
	m_plotToffs->legend->setFont(fontLegend);
	m_plotToffs->legend->setBrush(Qt::NoBrush);
	m_plotToffs->legend->setBorderPen(Qt::NoPen);
	m_plotToffs->setBackground(background);
	m_statsTEdit = new QPlainTextEdit;
	m_statsTEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_statsTEdit->setReadOnly(true);
	m_statsTEdit->setTextInteractionFlags(m_statsTEdit->textInteractionFlags() | Qt::TextSelectableByKeyboard);
	QVBoxLayout* layoutPlotsClean = new QVBoxLayout;
	layoutPlotsClean->addWidget(blinksPlotLbl);
	layoutPlotsClean->addWidget(m_plotBlinks);
	layoutPlotsClean->addWidget(tonsPlotLbl);
	layoutPlotsClean->addWidget(m_plotTOns);
	layoutPlotsClean->addWidget(toffsPlotLbl);
	layoutPlotsClean->addWidget(m_plotToffs);
	layoutPlotsClean->addWidget(m_statsTEdit);
	m_groupBoxCleanerPlots->setLayout(layoutPlotsClean);

	QVBoxLayout* layoutClean = new QVBoxLayout;
	layoutClean->addWidget(groupBoxCleanerParams);
	layoutClean->addWidget(m_groupBoxCleanerPlots);
	layoutClean->addWidget(emptyWidgetClean);
	m_cleanerWidget = new QWidget;
	m_cleanerWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_cleanerWidget->setLayout(layoutClean);
}

DetectionSetWidget::~DetectionSetWidget()
{

}

void DetectionSetWidget::actionNeeded()
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("DetectionSet");
	if (!bc) return;
	poca::core::CommandableObject* dset = dynamic_cast <poca::core::CommandableObject*>(bc);

	QObject* sender = QObject::sender();
	bool found = false;
	for (size_t n = 0; n < m_lutButtons.size() && !found; n++) {
		found = (m_lutButtons[n].first == sender);
		if (found) {
			dset->executeCommand(true, "changeLUT", "LUT", m_lutButtons[n].second);
			for (poca::plot::FilterHistogramWidget* histW : m_histWidgets)
				histW->redraw();
			m_object->notifyAll("updateDisplay");
			return;
		}
	}
	for (size_t n = 0; n < m_lutHeatmapButtons.size() && !found; n++) {
		found = (m_lutHeatmapButtons[n].first == sender);
		if (found) {
			dset->executeCommand(true, "changeLUTHeatmap", m_lutHeatmapButtons[n].second);
			m_object->notifyAll("updateDisplay");
			return;
		}
	}
	if (sender == m_maxRadiusEdit) {
		bool ok;
		int val = m_maxRadiusEdit->text().toInt(&ok);
		if (!ok) return;
		m_radiusSlider->blockSignals(true);
		m_radiusSlider->setMaximum(val);
		m_radiusSlider->blockSignals(false);
	}
	if (sender == m_currentRadiusEdit) {
		bool ok;
		float val = m_currentRadiusEdit->text().toFloat(&ok);
		if (!ok) return;
		m_radiusSlider->blockSignals(true);
		m_radiusSlider->setSliderPosition(val);
		m_radiusSlider->blockSignals(false);
		dset->executeCommand(true, "radiusHeatmap", val);
		m_object->notifyAll("updateDisplay");
	}
	if (sender == m_intensityEdit) {
		bool ok;
		float val = m_intensityEdit->text().toFloat(&ok);
		if (!ok) return;
		dset->executeCommand(true, "intensityHeatmap", val);
		m_object->notifyAll("updateDisplay");
	}
	if (sender == m_cleanButton) {
		bool ok;
		float radius = m_radiusCleanerEdit->text().toFloat(&ok);
		if (!ok) return;
		uint32_t maxDT = m_maxDarkTEdit->text().toUInt(&ok);
		if (!ok) return;
		bool fixedDT = m_fixedDarkTcbox->isChecked();
		poca::core::CommandInfo ci(true, "clean", "radius", radius, "maxDarkTime", maxDT, "fixedDarkTime", fixedDT);
		dset->executeCommand(&ci);
		if (ci.hasParameter("object")) {
			poca::core::MyObjectInterface* obj = ci.getParameterPtr<poca::core::MyObjectInterface>("object");
			emit(transferNewObjectCreated(obj));
		}
	}
	if (sender == m_saveFramesButton) {
		dset->executeCommand(true, "saveFramesMergedLocs");
	}
	if (sender == m_saveDetectionsButton) {
		QString name = m_object->getName().c_str(), filename(m_object->getDir().c_str());
		if(!filename.endsWith("/"))
			filename.append("/");
		filename.append(name);
		QString extension = name.indexOf(".") != -1 ? name.right(name.size() - name.indexOf(".")) : ".csv";
		filename = QFileDialog::getSaveFileName(NULL, QObject::tr("Save detections..."), filename, QString("Stats files (*" + extension + ")"), 0, QFileDialog::DontUseNativeDialog);
		if (filename.isEmpty()) return;
		dset->executeCommand(true, "saveLocalizations", "path", filename.toStdString());
	}
	if (sender == m_creationObjectsOnLabelsButton) {
		//m_object->executeCommandOnSpecificComponent("DetectionSet", &poca::core::CommandInfo(true, "createDBSCANObjects",
		//	"myObject", m_object));
		dset->executeCommand(true, "createObjectsOnLabels");
		m_object->notifyAll("updateDisplay");
	}
}

void DetectionSetWidget::actionNeeded(bool _val)
{
	if (m_object == NULL) return;
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("DetectionSet");
	if (!bc) return;
	poca::core::CommandableObject* dset = dynamic_cast <poca::core::CommandableObject*>(bc);

	QObject* sender = QObject::sender();
	if (sender == m_displayButton) {
		dset->executeCommand(true, "selected", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_pointRenderButton) {
		dset->executeCommand(true, "pointRendering", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_heatmapButton) {
		dset->executeCommand(true, "displayHeatmap", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_displayCleanButton) {
		dset->executeCommand(true, "displayCleanedLocs", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_radiusScreenHeatCbox) {
		dset->executeCommand(true, "radiusHeatmapType", "radiusScreenHeatmap", _val, "radiusWorldHeatmap", !_val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_radiusWorldHeatCbox) {
		dset->executeCommand(true, "radiusHeatmapType", "radiusScreenHeatmap", !_val, "radiusWorldHeatmap", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_interpolateLUTHeatmapCbox) {
		dset->executeCommand(true, "interpolateHeatmapLUT", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_gaussianButton) {
		dset->executeCommand(true, "displayGaussian", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_fixedSizeGaussCBox) {
		dset->executeCommand(true, "fixedSizeGaussian", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_worldButton) {
		dset->executeCommand(true, "screenCoordinates", !_val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_screenButton) {
		dset->executeCommand(true, "screenCoordinates", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
}

void DetectionSetWidget::actionNeeded(int _val)
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("DetectionSet");
	if (!bc) return;
	poca::core::CommandableObject* dset = dynamic_cast <poca::core::CommandableObject*>(bc);

	QObject* sender = QObject::sender();
	if (sender == m_radiusSlider) {
		m_currentRadiusEdit->blockSignals(true);
		m_currentRadiusEdit->setText(QString::number(_val));
		m_currentRadiusEdit->blockSignals(false);
		dset->executeCommand(true, "radiusHeatmap", (float)_val);
		m_object->notifyAll("updateDisplay");
	}
	if (sender == m_intensitySlider) {
		float val = (float)_val / 1000.f;
		m_intensityEdit->blockSignals(true);
		m_intensityEdit->setText(QString::number(val));
		m_intensityEdit->blockSignals(false);
		dset->executeCommand(true, "intensityHeatmap", val);
		m_object->notifyAll("updateDisplay");
	}
	if (sender == m_alphaGaussianSlider) {
		float val = (float)_val / 1000.f;
		m_alphaGaussianSlider->blockSignals(true);
		m_alphaValueLbl->setText("Alpha: " + QString::number(val));
		m_alphaGaussianSlider->blockSignals(false);
		dset->executeCommand(true, "alphaGaussian", val);
		m_object->notifyAll("updateDisplay");
	}
	if (sender == m_sizePointSpn) {
		unsigned int valD = this->pointSize();
		dset->executeCommand(true, "pointSizeGL", valD);
		m_object->notifyAll("updateDisplay");
	}
}

void DetectionSetWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
	if (_obj == NULL) {
		update(NULL, "");
		return;
	}
	poca::core::MyObjectInterface* obj = _obj->currentObject();
	bool actionDone = false;
	if (_ci->nameCommand == "histogram" || _ci->nameCommand == "changeLUT" || _ci->nameCommand == "selected") {
		if (_ci->nameCommand == "histogram") {
			std::string action = _ci->getParameter<std::string>("action");
			if (action == "save")
				_ci->addParameter("dir", _obj->getDir());
		}
		poca::core::BasicComponentInterface* bc = obj->getBasicComponent("DetectionSet");
		bc->executeCommand(_ci);
		actionDone = true;
	}
	if (actionDone) {
		_obj->notifyAll("LoadObjCharacteristicsDetectionSetWidget");
		_obj->notifyAll("updateDisplay");
	}
}

void DetectionSetWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	poca::core::MyObjectInterface* objOneColor = obj->currentObject();
	m_object = obj;

	bool visible = (objOneColor != NULL && objOneColor->hasBasicComponent("DetectionSet"));

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	auto index = m_parentTab->currentIndex();
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
	m_parentTab->setTabVisible(m_parentTab->indexOf(m_cleanerWidget), visible);
	m_parentTab->setCurrentIndex(index);
#endif

	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsDetectionSetWidget") {

		poca::core::BasicComponentInterface* bci = objOneColor->getBasicComponent("DetectionSet");
		if (!bci) return;
		poca::core::stringList nameData = bci->getNameData();

		QVBoxLayout* layout = NULL;
		//First time we load data -> no hist widget was created before
		if (m_histWidgets.empty()) {
			layout = new QVBoxLayout;
			layout->setContentsMargins(1, 1, 1, 1);
			layout->setSpacing(1);
			for (size_t n = 0; n < nameData.size(); n++) {
				m_histWidgets.push_back(new poca::plot::FilterHistogramWidget(m_mediator, "DetectionSet", this));
				layout->addWidget(m_histWidgets[n]);
			}
			m_detectionSetFilteringWidget->setLayout(layout);
		}
		else if (nameData.size() > m_histWidgets.size()) {
			layout = dynamic_cast <QVBoxLayout*>(m_detectionSetFilteringWidget->layout());
			//Here, we need to add some hist widgets because this loc data has more features than the one loaded before
			for (size_t n = m_histWidgets.size(); n < nameData.size(); n++) {
				m_histWidgets.push_back(new poca::plot::FilterHistogramWidget(m_mediator, "DetectionSet", this));
				layout->addWidget(m_histWidgets[n]);
			}
		}
		for (size_t n = 0; n < m_histWidgets.size(); n++)
			m_histWidgets[n]->setVisible(n < nameData.size());

		int cpt = 0;

		std::vector <float> ts, bins;
		for (std::string type : nameData) {
			poca::core::HistogramInterface* hist = bci->getHistogram(type);
			if (hist != NULL)
				m_histWidgets[cpt++]->setInfos(type.c_str(), hist, bci->isLogHistogram(type), bci->isCurrentHistogram(type) ? bci->getPalette() : NULL);
		}

		size_t nbLocs = bci->nbElements();
		m_nbLocsLbl->setText(QString("# localizations: %1").arg(nbLocs));


		m_detectionSetFilteringWidget->updateGeometry();

		bool selected = bci->isSelected();
		m_displayButton->blockSignals(true);
		m_displayButton->setChecked(selected);
		m_displayButton->blockSignals(false);

		if (m_object->nbColors() < 5 && bci->hasParameter("displayHeatmap")) {
			bool val = bci->getParameter<bool>("displayHeatmap");
			m_heatmapButton->setEnabled(true);
			m_heatmapButton->blockSignals(true);
			m_heatmapButton->setChecked(val);
			m_heatmapButton->blockSignals(false);

			m_groupBoxHeatmap->setVisible(true);
			if (bci->hasParameter("radiusHeatmap")) {
				float val = bci->getParameter<float>("radiusHeatmap");
				m_currentRadiusEdit->blockSignals(true);
				m_radiusSlider->blockSignals(true);
				m_currentRadiusEdit->setText(QString::number(val));
				m_radiusSlider->setSliderPosition(val);
				m_currentRadiusEdit->blockSignals(false);
				m_radiusSlider->blockSignals(false);
			}
			if (bci->hasParameter("intensityHeatmap")) {
				float val = bci->getParameter<float>("intensityHeatmap");
				m_intensityEdit->blockSignals(true);
				m_intensitySlider->blockSignals(true);
				m_intensityEdit->setText(QString::number(val));
				m_intensitySlider->setSliderPosition(val * 1000.f);
				m_intensityEdit->blockSignals(false);
				m_intensitySlider->blockSignals(false);
			}
			if (bci->hasParameter("radiusHeatmapType", "radiusScreenHeatmap")) {
				bool val = bci->getParameter<bool>("radiusHeatmapType", "radiusScreenHeatmap");
				m_radiusScreenHeatCbox->blockSignals(false);
				m_radiusScreenHeatCbox->setCheckable(val);
				m_radiusScreenHeatCbox->blockSignals(false);
			}
		}
		else {
			m_heatmapButton->blockSignals(true);
			m_heatmapButton->setChecked(false);
			m_heatmapButton->setEnabled(false);
			m_groupBoxHeatmap->setVisible(false);
			m_heatmapButton->blockSignals(false);
		}

		if (m_object->nbColors() < 5 && bci->hasParameter("displayGaussian")) {
			if (bci->hasParameter("alphaGaussian")) {
				float val = bci->getParameter<float>("alphaGaussian");
				m_alphaValueLbl->blockSignals(true);
				m_alphaGaussianSlider->blockSignals(true);
				m_alphaValueLbl->setText("Alpha: " + QString::number(val));
				m_alphaGaussianSlider->setSliderPosition(val * 1000.f);
				m_alphaValueLbl->blockSignals(false);
				m_alphaGaussianSlider->blockSignals(false);
			}
			if (bci->hasParameter("fixedSizeGaussian")) {
				bool val = bci->getParameter<bool>("fixedSizeGaussian");
				m_fixedSizeGaussCBox->blockSignals(false);
				m_fixedSizeGaussCBox->setChecked(val);
				m_fixedSizeGaussCBox->blockSignals(false);
			}
		}
		else {
			m_groupBoxGaussian->setEnabled(false);
			m_gaussianButton->blockSignals(true);
			m_gaussianButton->setChecked(false);
			m_gaussianButton->setEnabled(false);
			m_gaussianButton->blockSignals(false);
		}

		if (bci->hasParameter("displayCleanedLocs")) {
			bool val = bci->getParameter<bool>("displayCleanedLocs");
			m_displayCleanButton->blockSignals(true);
			m_displayCleanButton->setChecked(val);
			m_displayCleanButton->blockSignals(false);
		}

		if (bci->hasParameter("pointRendering")) {
			bool val = bci->getParameter<bool>("pointRendering");
			m_gaussianButton->blockSignals(true);
			m_pointRenderButton->blockSignals(true);
			m_pointRenderButton->setChecked(val);
			m_pointRenderButton->blockSignals(false);
			m_gaussianButton->blockSignals(false);
		}

		if (bci->hasParameter("pointSizeGL")) {
			uint32_t val = bci->getParameter<uint32_t>("pointSizeGL");
			m_sizePointSpn->blockSignals(true);
			m_sizePointSpn->setValue(val);
			m_sizePointSpn->blockSignals(false);
		}

		poca::core::CommandInfo ci(false, "DetectionSet", "getCleanEquations");
		m_object->executeCommand(&ci);
		if (ci.hasParameter("blinks")) {
			m_groupBoxCleanerPlots->setVisible(true);
			poca::core::EquationFit* eqnBlinks = ci.getParameterPtr<poca::core::EquationFit>("blinks");
			poca::core::EquationFit* eqnTons = ci.getParameterPtr<poca::core::EquationFit>("tons");
			poca::core::EquationFit* eqnToffs = ci.getParameterPtr<poca::core::EquationFit>("toffs");
			uint32_t nbEmissionBursts = ci.getParameter<uint32_t>("nbEmissionBursts");
			uint32_t nbOriginalLocs = ci.getParameter<uint32_t>("nbOriginalLocs");
			uint32_t nbSupressedLocs = ci.getParameter<uint32_t>("nbSupressedLocs");
			uint32_t nbAddedLocs = ci.getParameter<uint32_t>("nbAddedLocs");
			uint32_t nbUncorrectedLocs = ci.getParameter<uint32_t>("nbUncorrectedLocs");
			uint32_t darkTime = ci.getParameter<uint32_t>("darkTime");
			uint32_t nbTotalClean = nbUncorrectedLocs + nbAddedLocs;

			fillPlot(m_plotBlinks, eqnBlinks);
			fillPlot(m_plotTOns, eqnTons);
			fillPlot(m_plotToffs, eqnToffs);

			double blinkFit = eqnBlinks->getParams()[0];
			double tonFit = eqnTons->getParams()[1];
			double kd = blinkFit * tonFit;
			double kb = tonFit - kd;
			double nblink = 1 + (kd / kb);
			double controlNbMol = (double)nbEmissionBursts / nblink;
			double err = ((fabs(nbTotalClean - controlNbMol)) / nbTotalClean) * 100.;

			QString tau(0x03C4);
			QString statsCleaner = "k_d / ( k_d + k_b ) = " + QString::number(blinkFit) + "\nk_d + k_b = " + QString::number(tonFit) + "\nk_d = " + QString::number(kd) + "\nk_b = " + QString::number(kb) + "\nN_blinks = 1 + (k_d / k_b) = " + QString::number(nblink) + "\n# detections for " + tau + "_" + QString::number(darkTime) + " = " + QString::number(nbTotalClean);// +"\n\nControl by blinks (#emission burst / N_blinks):\n# emission burst = " + QString::number(nbEmissionBursts) + "\n# detections for control by blinks = " + QString::number(nbEmissionBursts) + " / " + QString::number(nblink) + " = " + QString::number(controlNbMol) + "\n\nNormalized difference: (" + QString::number(nbTotalClean) + " - " + QString::number(controlNbMol) + " ) / " + QString::number(nbTotalClean) + " = " + QString::number(err) + "%";

			m_statsTEdit->clear();
			m_statsTEdit->appendPlainText(statsCleaner);
		}
		else {
			m_groupBoxCleanerPlots->setVisible(false);
		}
	}
}

void DetectionSetWidget::fillPlot(QCustomPlot* _customplot, poca::core::EquationFit* _eqn)
{
	_customplot->clearGraphs();
	_customplot->clearPlottables();
	_customplot->clearItems();

	const std::vector <double>& ts = _eqn->getTs();
	const std::vector <double>& values = _eqn->getValues();
	const std::vector <double>& fittedValues = _eqn->getFitValues();
	QVector<double> x1(ts.size()), y1(ts.size()), y2(ts.size());
	float maxY = 0.;
	for (int i = 0; i < x1.size(); ++i)
	{
		x1[i] = ts[i];
		y1[i] = values[i];
		y2[i] = fittedValues[i];
		if (y1[i] > maxY) maxY = y1[i];
		if (y2[i] > maxY) maxY = y2[i];
	}

	QCPBars* bars1 = new QCPBars(_customplot->xAxis, _customplot->yAxis);
	bars1->setWidth(9 / (double)x1.size());
	bars1->setData(x1, y1);
	bars1->setPen(Qt::NoPen);
	bars1->setBrush(QColor(10, 140, 70, 160));

	QCPGraph* graph2 = _customplot->addGraph();
	graph2->setData(x1, y2);
	graph2->setPen(QPen(Qt::red));
	graph2->setBrush(QColor(200, 200, 200, 20));

	// move bars above graphs and grid below bars:
	_customplot->addLayer("abovemain", _customplot->layer("main"), QCustomPlot::limAbove);
	_customplot->addLayer("belowmain", _customplot->layer("main"), QCustomPlot::limBelow);
	graph2->setLayer("abovemain");
	_customplot->xAxis->grid()->setLayer("belowmain");
	_customplot->yAxis->grid()->setLayer("belowmain");

	_customplot->xAxis->setRange(x1[0], x1[x1.size() - 1]);
	_customplot->yAxis->setRange(0, maxY);

	_customplot->setBackground(QColor(253, 253, 253));

	QCPItemText* nameHisto = new QCPItemText(_customplot);
	nameHisto->position->setType(QCPItemPosition::ptViewportRatio);
	nameHisto->position->setCoords(0.95, 0.15); // move 10 pixels to the top from bracket center anchor
	nameHisto->setPositionAlignment(Qt::AlignTop | Qt::AlignRight);
	nameHisto->setColor(Qt::black);
	nameHisto->setText(_eqn->getEquation().c_str());
	nameHisto->setFont(QFont(font().family(), 10));

	_customplot->legend->clearItems();
	_customplot->replot();
}

void DetectionSetWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

