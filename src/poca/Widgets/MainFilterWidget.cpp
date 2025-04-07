/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MainFilterWidget.cpp
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

#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QColorDialog>
#include <QtGui/QRegExpValidator>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QOpenGLWidget>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFileDialog>
#include <fstream>
#include <tuple>
#include <glm/gtx/string_cast.hpp>

#include <OpenGL/Camera.hpp>
#include <General/Misc.h>
#include <OpenGL/SsaoShader.hpp>
#include <Plot/Icons.hpp>

#include "../Widgets/MainFilterWidget.hpp"
#include "../Objects/SMLM_Object/SMLMObject.hpp"

MainFilterWidget::MainFilterWidget(poca::core::MediatorWObjectFWidget * _mediator, QWidget * _parent/*= 0*/, Qt::WindowFlags _f/*= 0 */) :QWidget(_parent, _f), m_firstIndexObj(1)
{
	m_mediator = _mediator;
	m_object = NULL;
	m_currentCamera = NULL;

	this->setObjectName( "MainFilterWidget" );
	this->addActionToObserve( "LoadObjCharacteristicsAllWidgets" );

	m_dockGeneral = new QDockWidget("General");
	m_dockGeneral->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
	m_dockGeneral->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_smoothPointCB = new QCheckBox( "Smooth point" );
	m_smoothPointCB->setChecked( true );
	m_smoothPointCB->setSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum );
	m_sizePointLbl = new QLabel( "Size point [1-100]:" );
	m_sizePointLbl->setSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum );
	m_sizePointSpn = new QSpinBox;
	m_sizePointSpn->setRange( 1, 100 );
	m_sizePointSpn->setValue( 1 );
	m_sizePointSpn->setSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum );
	m_smoothLineCB = new QCheckBox( "Smooth line" );
	m_smoothLineCB->setChecked( false );
	m_smoothLineCB->setSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum );
	m_widthLineLbl = new QLabel( "Line width [1-100]:" );
	m_widthLineLbl->setSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum );
	m_widthLineSpn = new QSpinBox;
	m_widthLineSpn->setRange( 1, 100 );
	m_widthLineSpn->setValue( 1 );
	m_widthLineSpn->setSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum );
	m_backColorLbl = new QLabel( "Background color:" );
	m_backColorLbl->setSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum );
	m_colorBackBtn = new QPushButton();
	m_colorBackBtn->setSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum );
	m_colorBackBtn->setStyleSheet( "background-color: rgb(255, 255, 255);"
		"border-style: outset;"
		"border-width: 2px;"
		"border-radius: 5px;"
		"border-color: black;"
		"font: 12px;"
		"min-width: 5em;"
		"padding: 3px;"
		);
	m_antialiasCB = new QCheckBox("Antialias");
	m_antialiasCB->setChecked(true);
	m_antialiasCB->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_fontSizeLbl = new QLabel("Font size [1-100]:");
	m_fontSizeLbl->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_fontSizeSpn = new QSpinBox;
	m_fontSizeSpn->setRange(1, 100);
	m_fontSizeSpn->setValue(20);
	m_fontSizeSpn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cullFaceCB = new QCheckBox("Cull face");
	m_cullFaceCB->setChecked(true);
	m_cullFaceCB->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_clipCB = new QCheckBox("Clip");
	m_clipCB->setChecked(true);
	m_clipCB->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_fillPolygonCB = new QCheckBox("Fill Polygon");
	m_fillPolygonCB->setChecked(true);
	m_fillPolygonCB->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_listCommandsBtn = new QPushButton("List commands");
	m_listCommandsBtn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QGridLayout * layoutMisc = new QGridLayout;
	int lineCount = 0, columnCount = 0;
	layoutMisc->addWidget( m_sizePointLbl, lineCount, columnCount++, 1, 1 );
	layoutMisc->addWidget( m_sizePointSpn, lineCount, columnCount++, 1, 1 );
	layoutMisc->addWidget( m_widthLineLbl, lineCount, columnCount++, 1, 1 );
	layoutMisc->addWidget( m_widthLineSpn, lineCount++, columnCount++, 1, 1 );
	columnCount = 0;
	layoutMisc->addWidget( m_smoothPointCB, lineCount, columnCount++, 1, 1 );
	layoutMisc->addWidget( m_smoothLineCB, lineCount, columnCount++, 1, 1 );
	layoutMisc->addWidget( m_backColorLbl, lineCount, columnCount++, 1, 1 );
	layoutMisc->addWidget( m_colorBackBtn, lineCount++, columnCount++, 1, 1 );
	columnCount = 0;
	layoutMisc->addWidget(m_antialiasCB, lineCount, columnCount++, 1, 1);
	layoutMisc->addWidget(m_fontSizeLbl, lineCount, columnCount++, 1, 1);
	layoutMisc->addWidget(m_fontSizeSpn, lineCount++, columnCount++, 1, 1);
	columnCount = 0;
	layoutMisc->addWidget(m_cullFaceCB, lineCount, columnCount++, 1, 1);
	layoutMisc->addWidget(m_clipCB, lineCount, columnCount++, 1, 1);
	layoutMisc->addWidget(m_fillPolygonCB, lineCount++, columnCount++, 1, 1);
	layoutMisc->addWidget(m_listCommandsBtn, lineCount, columnCount++, 1, 1);
	QWidget * generalW = new QWidget;
	generalW->setLayout(layoutMisc);
	m_dockGeneral->setWidget(generalW);

	QRegExp rx("[0-9.]+");
	QValidator* validator = new QRegExpValidator(rx, this);
	lineCount = columnCount = 0;
	m_dockInfoDataset = new QDockWidget("Dataset");
	m_dockInfoDataset->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
	m_dockInfoDataset->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	m_nameDatasetLbl = new QLabel("Name:");
	m_nameDatasetLbl->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	QLabel* widthDatasetLbl = new QLabel("Width:");
	widthDatasetLbl->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	m_lineEditWidthData = new QLineEdit;
	m_lineEditWidthData->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	m_lineEditWidthData->setValidator(validator);
	QLabel* heightDatasetLbl = new QLabel("Height:");
	heightDatasetLbl->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	m_lineEditHeightData = new QLineEdit;
	m_lineEditHeightData->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	m_lineEditHeightData->setValidator(validator);
	m_idDatasetLbl = new QLabel("Id:");
	m_idDatasetLbl->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	QGridLayout* layoutInfoDataset = new QGridLayout;
	layoutInfoDataset->addWidget(m_nameDatasetLbl, lineCount++, 0, 1, 5);
	layoutInfoDataset->addWidget(widthDatasetLbl, lineCount, columnCount++, 1, 1);
	layoutInfoDataset->addWidget(m_lineEditWidthData, lineCount, columnCount++, 1, 1);
	layoutInfoDataset->addWidget(heightDatasetLbl, lineCount, columnCount++, 1, 1);
	layoutInfoDataset->addWidget(m_lineEditHeightData, lineCount, columnCount++, 1, 1);
	layoutInfoDataset->addWidget(m_idDatasetLbl, lineCount++, columnCount++, 1, 1);
	QWidget* groupInfoW = new QWidget;
	groupInfoW->setLayout(layoutInfoDataset);
	m_dockInfoDataset->setWidget(groupInfoW);
	m_dockInfoDataset->setVisible(false);

	m_dockGrid = new QDockWidget("Grid");
	m_dockGrid->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
	m_dockGrid->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxSameAllDimGrid = new QCheckBox("Isotropic definition");
	m_cboxSameAllDimGrid->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxSameAllDimGrid->setChecked(true);
	m_cboxNbGrid = new QCheckBox("Number definition");
	m_cboxNbGrid->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxNbGrid->setChecked(true);
	m_cboxLengthGrid = new QCheckBox("Length definition");
	m_cboxLengthGrid->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxLengthGrid->setChecked(false);
	QButtonGroup* bgroupGrid = new QButtonGroup;
	bgroupGrid->addButton(m_cboxNbGrid);
	bgroupGrid->addButton(m_cboxLengthGrid);
	QLabel* dimXNb = new QLabel("X:");
	dimXNb->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_leditDimXNb = new QLineEdit("5");
	m_leditDimXNb->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* dimYNb = new QLabel("Y:");
	dimYNb->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_leditDimYNb = new QLineEdit("5");
	m_leditDimYNb->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* dimZNb = new QLabel("Z:");
	dimZNb->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_leditDimZNb = new QLineEdit("5");
	m_leditDimZNb->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* dimXLength = new QLabel("X:");
	dimXLength->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_leditDimXLength = new QLineEdit("50");
	m_leditDimXLength->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* dimYLength = new QLabel("Y:");
	dimYLength->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_leditDimYLength = new QLineEdit("50");
	m_leditDimYLength->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* dimZLength = new QLabel("Z:");
	dimZLength->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_leditDimZLength = new QLineEdit("50");
	m_leditDimZLength->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QFrame* separator = new QFrame(this);
	separator->setLineWidth(1);
	separator->setMidLineWidth(1);
	separator->setFrameShape(QFrame::HLine);
	separator->setPalette(QPalette(QColor(0, 0, 0)));
	QHBoxLayout* layoutGridL1 = new QHBoxLayout;
	layoutGridL1->addWidget(m_cboxSameAllDimGrid, Qt::AlignRight);
	QHBoxLayout* layoutGridL2 = new QHBoxLayout;
	layoutGridL2->addWidget(m_cboxNbGrid, Qt::AlignCenter);
	layoutGridL2->addWidget(m_cboxLengthGrid, Qt::AlignCenter);
	QGridLayout* layoutGridNb = new QGridLayout;
	layoutGridNb->addWidget(dimXNb, 0, 0, 1, 1);
	layoutGridNb->addWidget(m_leditDimXNb, 0, 1, 1, 1);
	layoutGridNb->addWidget(dimYNb, 1, 0, 1, 1);
	layoutGridNb->addWidget(m_leditDimYNb, 1, 1, 1, 1);
	layoutGridNb->addWidget(dimZNb, 2, 0, 1, 1);
	layoutGridNb->addWidget(m_leditDimZNb, 2, 1, 1, 1);
	QGridLayout* layoutGridLength = new QGridLayout;
	layoutGridLength->addWidget(dimXLength, 0, 0, 1, 1);
	layoutGridLength->addWidget(m_leditDimXLength, 0, 1, 1, 1);
	layoutGridLength->addWidget(dimYLength, 1, 0, 1, 1);
	layoutGridLength->addWidget(m_leditDimYLength, 1, 1, 1, 1);
	layoutGridLength->addWidget(dimZLength, 2, 0, 1, 1);
	layoutGridLength->addWidget(m_leditDimZLength, 2, 1, 1, 1);
	QHBoxLayout* layoutGridInfos = new QHBoxLayout;
	layoutGridInfos->addLayout(layoutGridNb);
	layoutGridInfos->addLayout(layoutGridLength);
	QVBoxLayout* layoutGridComplet = new QVBoxLayout;
	layoutGridComplet->addLayout(layoutGridL1, Qt::AlignRight);
	layoutGridComplet->addLayout(layoutGridL2);
	layoutGridComplet->addLayout(layoutGridInfos);
	QWidget* gridW = new QWidget;
	gridW->setLayout(layoutGridComplet);
	m_dockGrid->setWidget(gridW);

	/************************************************************************/
	/* For SSAO                                                             */
	/************************************************************************/
	m_dockSSAO = new QDockWidget("SSAO");
	m_dockSSAO->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
	m_dockSSAO->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxUseSSAO = new QCheckBox("SSAO");
	m_cboxUseSSAO->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxUseSSAO->setChecked(false);
	m_cboxSSAOSilhouette = new QCheckBox("Silhouette");
	m_cboxSSAOSilhouette->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxSSAOSilhouette->setChecked(false);
	m_cboxSSAOUseDebug = new QCheckBox("Use debug");
	m_cboxSSAOUseDebug->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxSSAOUseDebug->setChecked(false);
	m_cboxSSAODisplayPos = new QCheckBox("Pos");
	m_cboxSSAODisplayPos->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxSSAODisplayPos->setChecked(false);
	m_cboxSSAODisplayNormal = new QCheckBox("Normal");
	m_cboxSSAODisplayNormal->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxSSAODisplayNormal->setChecked(false);
	m_cboxSSAODisplayColor = new QCheckBox("Color");
	m_cboxSSAODisplayColor->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxSSAODisplayColor->setChecked(false);
	m_cboxSSAODisplaySSAOMap = new QCheckBox("SSAO map");
	m_cboxSSAODisplaySSAOMap->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxSSAODisplaySSAOMap->setChecked(false);
	QButtonGroup* bgroupSSAO = new QButtonGroup;
	bgroupSSAO->addButton(m_cboxSSAODisplayPos);
	bgroupSSAO->addButton(m_cboxSSAODisplayNormal);
	bgroupSSAO->addButton(m_cboxSSAODisplayColor);
	bgroupSSAO->addButton(m_cboxSSAODisplaySSAOMap);
	m_lblRadiusSSAO = new QLabel("Radius SSAO (pixel):");
	m_lblRadiusSSAO->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Expanding);
	m_leditRadiusSSAO = new QLineEdit("");
	m_leditRadiusSSAO->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Expanding);
	m_sliderRadiusSSAO = new QSlider;
	m_sliderRadiusSSAO->setOrientation(Qt::Horizontal);
	m_sliderRadiusSSAO->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_sliderRadiusSSAO->setRange(0, 100);
	m_lblStrengthSSAO = new QLabel("Strength SSAO:");
	m_lblStrengthSSAO->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Expanding);
	m_leditStrenghtSSAO = new QLineEdit("");
	m_leditStrenghtSSAO->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Expanding);
	m_sliderStrengthSSAO = new QSlider;
	m_sliderStrengthSSAO->setOrientation(Qt::Horizontal);
	m_sliderStrengthSSAO->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_sliderStrengthSSAO->setRange(0, 100);
	QGridLayout* layoutSSAO = new QGridLayout;
	lineCount = 0; columnCount = 0;
	layoutSSAO->addWidget(m_cboxUseSSAO, 0, columnCount++, 1, 1);
	layoutSSAO->addWidget(m_cboxSSAOSilhouette, 0, columnCount++, 1, 1);
	layoutSSAO->addWidget(m_cboxSSAOUseDebug, 0, columnCount++, 1, 1);
	columnCount = 0;
	layoutSSAO->addWidget(m_cboxSSAODisplayPos, 1, columnCount++, 1, 1);
	layoutSSAO->addWidget(m_cboxSSAODisplayNormal, 1, columnCount++, 1, 1);
	layoutSSAO->addWidget(m_cboxSSAODisplayColor, 1, columnCount++, 1, 1);
	layoutSSAO->addWidget(m_cboxSSAODisplaySSAOMap, 1, columnCount++, 1, 1);
	columnCount = 0;
	layoutSSAO->addWidget(m_lblRadiusSSAO, 2, columnCount++, 1, 1);
	layoutSSAO->addWidget(m_leditRadiusSSAO, 2, columnCount++, 1, 1);
	layoutSSAO->addWidget(m_sliderRadiusSSAO, 2, columnCount++, 1, 4);
	columnCount = 0;
	layoutSSAO->addWidget(m_lblStrengthSSAO, 3, columnCount++, 1, 1);
	layoutSSAO->addWidget(m_leditStrenghtSSAO, 3, columnCount++, 1, 1);
	layoutSSAO->addWidget(m_sliderStrengthSSAO, 3, columnCount++, 1, 4);
	QWidget* ssaoW = new QWidget;
	ssaoW->setLayout(layoutSSAO);
	m_dockSSAO->setWidget(ssaoW);

	/************************************************************************/
	/* For Camera position                                                      */
	/************************************************************************/
	columnCount = 0;
	m_dockCameraPosition = new QDockWidget("Camera position");
	m_dockCameraPosition->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
	m_dockCameraPosition->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_savePositionBtn = new QPushButton("Save camera pos");
	m_savePositionBtn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_loadPositionBtn = new QPushButton("Load camera pos");
	m_loadPositionBtn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxViewCamera = new QCheckBox("View");
	m_cboxViewCamera->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxViewCamera->setChecked(true);
	m_cboxRotationCamera = new QCheckBox("Rotation");
	m_cboxRotationCamera->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxRotationCamera->setChecked(true);
	m_cboxTranslationCamera = new QCheckBox("Translation");
	m_cboxTranslationCamera->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxTranslationCamera->setChecked(true);
	m_cboxZoomCamera = new QCheckBox("Zoom");
	m_cboxZoomCamera->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxZoomCamera->setChecked(true);
	m_cboxCropCamera = new QCheckBox("Crop");
	m_cboxCropCamera->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_cboxCropCamera->setChecked(true);

	QGridLayout* layoutCameraPosition = new QGridLayout;
	lineCount = 0; columnCount = 0;
	layoutCameraPosition->addWidget(m_savePositionBtn, lineCount, columnCount++, 1, 1);
	layoutCameraPosition->addWidget(m_loadPositionBtn, lineCount++, columnCount++, 1, 1);
	columnCount = 0;
	layoutCameraPosition->addWidget(m_cboxViewCamera, lineCount, columnCount++, 1, 1);
	layoutCameraPosition->addWidget(m_cboxRotationCamera, lineCount, columnCount++, 1, 1);
	layoutCameraPosition->addWidget(m_cboxTranslationCamera, lineCount++, columnCount++, 1, 1);
	columnCount = 0;
	layoutCameraPosition->addWidget(m_cboxZoomCamera, lineCount, columnCount++, 1, 1);
	layoutCameraPosition->addWidget(m_cboxCropCamera, lineCount, columnCount++, 1, 1);
	QWidget* cameraPositionW = new QWidget;
	cameraPositionW->setLayout(layoutCameraPosition);
	m_dockCameraPosition->setWidget(cameraPositionW);

	m_emptyWidget = new QWidget;
	m_emptyWidget->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Expanding);

	QVBoxLayout* layout = new QVBoxLayout;
	layout->addWidget(m_dockGeneral);
	layout->addWidget(m_dockInfoDataset);
	layout->addWidget(m_dockGrid);
	layout->addWidget(m_dockSSAO);
	layout->addWidget(m_dockCameraPosition);
	layout->addWidget(m_emptyWidget);

	this->setLayout( layout );
	this->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Expanding);

	QObject::connect( m_sizePointSpn, SIGNAL( valueChanged ( int ) ), this, SLOT( actionNeeded( int ) ) );
	QObject::connect( m_smoothPointCB, SIGNAL( toggled( bool ) ), this, SLOT( actionNeeded( bool ) ) );
	QObject::connect( m_widthLineSpn, SIGNAL( valueChanged ( int ) ), this, SLOT( actionNeeded( int ) ) );
	QObject::connect( m_smoothLineCB, SIGNAL( toggled( bool ) ), this, SLOT( actionNeeded( bool ) ) );
	QObject::connect(m_fontSizeSpn, SIGNAL(valueChanged(int)), this, SLOT(actionNeeded(int)));
	QObject::connect( m_colorBackBtn, SIGNAL( clicked() ), this, SLOT( actionNeeded() ) );
	QObject::connect(m_antialiasCB, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_cullFaceCB, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_clipCB, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_fillPolygonCB, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_savePositionBtn, SIGNAL(clicked()), this, SLOT(actionNeeded()));
	QObject::connect(m_loadPositionBtn, SIGNAL(clicked()), this, SLOT(actionNeeded()));
	QObject::connect(m_listCommandsBtn, SIGNAL(clicked()), this, SLOT(actionNeeded()));

	QObject::connect(m_lineEditWidthData, SIGNAL(editingFinished()), this, SLOT(actionNeeded())); 
	QObject::connect(m_lineEditHeightData, SIGNAL(editingFinished()), this, SLOT(actionNeeded())); 

	QObject::connect(m_leditDimXNb, SIGNAL(editingFinished()), this, SLOT(actionNeeded()));
	QObject::connect(m_leditDimYNb, SIGNAL(editingFinished()), this, SLOT(actionNeeded()));
	QObject::connect(m_leditDimZNb, SIGNAL(editingFinished()), this, SLOT(actionNeeded()));
	QObject::connect(m_leditDimXLength, SIGNAL(editingFinished()), this, SLOT(actionNeeded()));
	QObject::connect(m_leditDimYLength, SIGNAL(editingFinished()), this, SLOT(actionNeeded()));
	QObject::connect(m_leditDimZLength, SIGNAL(editingFinished()), this, SLOT(actionNeeded()));
	QObject::connect(m_cboxSameAllDimGrid, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_cboxNbGrid, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_cboxLengthGrid, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));

	QObject::connect(m_cboxUseSSAO, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_cboxSSAOSilhouette, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_cboxSSAOUseDebug, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_cboxSSAODisplayPos, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_cboxSSAODisplayNormal, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_cboxSSAODisplayColor, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_cboxSSAODisplaySSAOMap, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_leditRadiusSSAO, SIGNAL(returnPressed()), this, SLOT(actionNeeded()));
	QObject::connect(m_sliderRadiusSSAO, SIGNAL(valueChanged(int)), this, SLOT(actionNeeded(int))); 
	QObject::connect(m_leditStrenghtSSAO, SIGNAL(returnPressed()), this, SLOT(actionNeeded()));
	QObject::connect(m_sliderStrengthSSAO, SIGNAL(valueChanged(int)), this, SLOT(actionNeeded(int)));

}

MainFilterWidget::~MainFilterWidget()
{
}

void MainFilterWidget::actionNeeded()
{
	if (m_object == NULL) return;
	QObject * sender = QObject::sender();
	if (sender == m_colorBackBtn){
		QColor color = QColorDialog::getColor();
		if (color.isValid()){
			this->setColorBackgroundButton(color.red(), color.green(), color.blue());
			m_object->executeCommand(&poca::core::CommandInfo(true, "colorBakground", 
				std::array <unsigned char, 4>{ (unsigned char)color.red(), 
												(unsigned char)color.green(),
												(unsigned char)color.blue(), 
												(unsigned char)255 }));
		}
	}
	else if (sender == m_setFirstImageIdBtn){
		bool ok;
		unsigned int tmp = m_indexFirstImageEdit->text().toUInt(&ok);
		if (ok)
			m_object->executeCommand(&poca::core::CommandInfo(true, "setImageIndex", "id", tmp));
	}
	else if (sender == m_lineEditWidthData){
		bool ok;
		double val = m_lineEditWidthData->text().toDouble(&ok);
		if (ok)
			m_object->executeCommand(&poca::core::CommandInfo(true, "widthDataset", val));
	}
	else if (sender == m_lineEditHeightData){
		bool ok;
		double val = m_lineEditHeightData->text().toDouble(&ok);
		if (ok)
			m_object->executeCommand(&poca::core::CommandInfo(true, "heightDataset", val));
	}

	else if (sender == m_leditDimXNb || sender == m_leditDimYNb || sender == m_leditDimZNb) {
		QLineEdit* tmp = static_cast <QLineEdit*>(sender);
		if (tmp) {
			bool ok;
			uint8_t val = tmp->text().toUInt(&ok);
			if (!ok) return;
			if (m_cboxSameAllDimGrid->isChecked()) {
				m_leditDimXNb->blockSignals(true);
				m_leditDimYNb->blockSignals(true);
				m_leditDimZNb->blockSignals(true);
				m_leditDimXNb->setText(QString::number(val));
				m_leditDimYNb->setText(QString::number(val));
				m_leditDimZNb->setText(QString::number(val));
				m_leditDimXNb->blockSignals(false);
				m_leditDimYNb->blockSignals(false);
				m_leditDimZNb->blockSignals(false);
			}
			else {
				tmp->blockSignals(true);
				tmp->setText(QString::number(val));
				tmp->blockSignals(false);
			}
		}
		bool ok;
		uint8_t valX = m_leditDimXNb->text().toUInt(&ok);
		if (!ok) return;
		uint8_t valY = m_leditDimYNb->text().toUInt(&ok);
		if (!ok) return;
		uint8_t valZ = m_leditDimZNb->text().toUInt(&ok);
		if (!ok) return;
		m_object->executeCommand(&poca::core::CommandInfo(true, "nbGrid", std::array <uint8_t, 3>{ valX, valY, valZ }));
		if(m_object != NULL)
			m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_leditDimXLength || sender == m_leditDimYLength || sender == m_leditDimZLength) {
		QLineEdit* tmp = static_cast <QLineEdit*>(sender);
		if (tmp) {
			bool ok;
			float val = tmp->text().toFloat(&ok);
			if (!ok) return;
			if (m_cboxSameAllDimGrid->isChecked()) {
				m_leditDimXLength->blockSignals(true);
				m_leditDimYLength->blockSignals(true);
				m_leditDimZLength->blockSignals(true);
				m_leditDimXLength->setText(QString::number(val));
				m_leditDimYLength->setText(QString::number(val));
				m_leditDimZLength->setText(QString::number(val));
				m_leditDimXLength->blockSignals(false);
				m_leditDimYLength->blockSignals(false);
				m_leditDimZLength->blockSignals(false);
			}
			else{
				tmp->blockSignals(true);
				tmp->setText(QString::number(val));
				tmp->blockSignals(false);
			}
		}
		bool ok;
		float valX = m_leditDimXLength->text().toFloat(&ok);
		if (!ok) return;
		float valY = m_leditDimYLength->text().toFloat(&ok);
		if (!ok) return;
		float valZ = m_leditDimZLength->text().toFloat(&ok);
		if (!ok) return;
		m_object->executeCommand(&poca::core::CommandInfo(true, "stepGrid", std::array <float, 3>{ valX, valY, valZ }));
		if (m_object != NULL)
			m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_savePositionBtn) {
		//emit(savePosition(QString()));
		emit(getCurrentCamera());
		if (m_currentCamera != NULL) {
			QString filename("cameraPosition.json");
			filename = QFileDialog::getSaveFileName(NULL, QObject::tr("Save camera position..."), filename, QString("json files (*.json)"), 0, QFileDialog::DontUseNativeDialog);
			if (filename.isEmpty()) return;
			const poca::opengl::StateCamera& stateCam = m_currentCamera->getStateCamera();
			nlohmann::json json;
			json["stateCamera"]["matrixView"] = stateCam.m_matrixView;
			json["stateCamera"]["rotationSum"] = stateCam.m_rotationSum;
			json["stateCamera"]["rotation"] = stateCam.m_rotation;
			json["stateCamera"]["center"] = stateCam.m_center;
			json["stateCamera"]["eye"] = stateCam.m_eye;
			json["stateCamera"]["matrix"] = stateCam.m_matrix;
			json["stateCamera"]["up"] = stateCam.m_up;
			json["stateCamera"]["translationModel"] = m_currentCamera->getTranslationModel();
			json["distanceOrtho"] = m_currentCamera->getDistanceOrtho();
			json["distanceOrthoOriginal"] = m_currentCamera->getOriginalDistanceOrtho();
			json["crop"] = m_currentCamera->getCurrentCrop();

			std::string text = json.dump();
			std::cout << text << std::endl;
			std::ofstream fs(filename.toLatin1().data());
			fs << text;
			fs.close();
		}
	}
	else if (sender == m_loadPositionBtn) {
		emit(loadPosition(QString()));
	}
	else if (sender == m_listCommandsBtn) {
		for (poca::core::BasicComponentInterface* bc : m_object->getComponents()) {
			std::cout << "For component " << bc->name() << ":" << std::endl;
			for (poca::core::Command* com : bc->getCommands()) {
				std::cout << "      -> " << com->name() << std::endl;
			}
		}
	}
	else if (sender == m_leditRadiusSSAO) {
		bool ok;
		double val = m_leditRadiusSSAO->text().toDouble(&ok);
		if (ok) {
			m_object->executeCommand(&poca::core::CommandInfo(true, "radiusSSAO", val));
			m_object->notifyAll("updateDisplay");

			m_sliderRadiusSSAO->blockSignals(true);
			m_sliderRadiusSSAO->setValue(val);
			m_sliderRadiusSSAO->blockSignals(false);
		}
	}
	else if (sender == m_leditStrenghtSSAO) {
		bool ok;
		double val = m_leditStrenghtSSAO->text().toDouble(&ok);
		if (ok) {
			m_object->executeCommand(&poca::core::CommandInfo(true, "strengthSSAO", val));
			m_object->notifyAll("updateDisplay");

			if (val < 0.f) {
				val = 0.05f;
				m_leditStrenghtSSAO->blockSignals(true);
				m_leditStrenghtSSAO->setText(QString::number(val));
				m_leditStrenghtSSAO->blockSignals(false);
			}
			if (val < 5.f) {
				val = 5.f;
				m_leditStrenghtSSAO->blockSignals(true);
				m_leditStrenghtSSAO->setText(QString::number(val));
				m_leditStrenghtSSAO->blockSignals(false);
			}
			float step = 0.05f;
			int tick = (int)(val / step);
			m_sliderStrengthSSAO->blockSignals(true);
			m_sliderStrengthSSAO->setValue(tick);
			m_sliderStrengthSSAO->blockSignals(false);
		}
	}
	/*else if (sender == m_btnPosition1) {
		QString path = QDir::currentPath();
		QString filename = QFileDialog::getOpenFileName(0,
			QObject::tr("Select one camera position file to open"),
			path,
			QObject::tr("Camera position file (*.json)"), 0, QFileDialog::DontUseNativeDialog);

		if (filename.isEmpty())
			return;

		m_filePath1 = filename;
		filename = filename.right(filename.size() - (filename.lastIndexOf("/") + 1));
		m_lblPosition1->setText(filename);
	}
	else if (sender == m_btnPosition2) {
		QString path = QDir::currentPath();
		QString filename = QFileDialog::getOpenFileName(0,
			QObject::tr("Select one camera position file to open"),
			path,
			QObject::tr("Camera position file (*.json)"), 0, QFileDialog::DontUseNativeDialog);

		if (filename.isEmpty())
			return;

		m_filePath2 = filename;
		filename = filename.right(filename.size() - (filename.lastIndexOf("/") + 1));
		m_lblPosition2->setText(filename);
	}
	else if (sender == m_btnApplyPosition1) {
		if (m_lblPosition1->text().isEmpty())
			return;
		emit(loadPosition(m_filePath1));
	}
	else if (sender == m_btnApplyPosition2) {
		if (m_lblPosition2->text().isEmpty())
			return;
		emit(loadPosition(m_filePath2));
	}*/
	
}

void MainFilterWidget::actionNeeded( int _val )
{
	if (m_object == NULL) return;
	QObject * sender = QObject::sender();
	if (sender == m_sizePointSpn){
		unsigned int valD = this->pointSize();
		m_object->executeCommand(&poca::core::CommandInfo(true, "pointSizeGL", valD));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_widthLineSpn){
		unsigned int valD = this->lineWidth();
		m_object->executeCommand(&poca::core::CommandInfo(true, "lineWidthGL", valD));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_fontSizeSpn) {
		float valD = this->fontSize();
		m_object->executeCommand(&poca::core::CommandInfo(true, "fontSize", valD));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_sliderRadiusSSAO) {
		float radius = this->radiusSSAO();
		m_leditRadiusSSAO->blockSignals(true);
		m_leditRadiusSSAO->setText(QString::number(radius));
		m_leditRadiusSSAO->blockSignals(false);
		m_object->executeCommand(&poca::core::CommandInfo(true, "radiusSSAO", radius));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_sliderStrengthSSAO) {
		float strength = this->strengthSSAO() * 0.05f;
		m_leditStrenghtSSAO->blockSignals(true);
		m_leditStrenghtSSAO->setText(QString::number(strength));
		m_leditStrenghtSSAO->blockSignals(false);
		m_object->executeCommand(&poca::core::CommandInfo(true, "strengthSSAO", strength));
		m_object->notifyAll("updateDisplay");
	}

}

void MainFilterWidget::actionNeeded(bool _val)
{
	if (m_object == NULL) return;
	QObject* sender = QObject::sender();
	if (sender == m_smoothPointCB)
		m_object->executeCommand(&poca::core::CommandInfo(true, "smoothPoint", this->isPointSmooth()));
	else if (sender == m_smoothLineCB)
		m_object->executeCommand(&poca::core::CommandInfo(true, "smoothLine", this->isLineSmooth()));
	else if (sender == m_cboxNbGrid || sender == m_cboxLengthGrid) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "useNbForGrid", m_cboxNbGrid->isChecked()));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_cboxSameAllDimGrid) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "isotropicGrid", m_cboxSameAllDimGrid->isChecked()));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_antialiasCB) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "antialias", m_antialiasCB->isChecked()));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_cullFaceCB) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "cullFace", m_cullFaceCB->isChecked()));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_clipCB) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "clip", m_clipCB->isChecked()));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_fillPolygonCB) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "fillPolygon", m_fillPolygonCB->isChecked()));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_cboxUseSSAO) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "useSSAO", m_cboxUseSSAO->isChecked()));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_cboxSSAOSilhouette) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "useSilhouetteSSAO", m_cboxSSAOSilhouette->isChecked()));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_cboxSSAOUseDebug) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "useDebugSSAO", m_cboxSSAOUseDebug->isChecked()));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_cboxSSAODisplayPos) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "currentDebugSSAO", poca::opengl::SsaoShader::SSAODebugDisplay::SSAO_POS));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_cboxSSAODisplayNormal) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "currentDebugSSAO", poca::opengl::SsaoShader::SSAODebugDisplay::SSAO_NORMAL));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_cboxSSAODisplayColor) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "currentDebugSSAO", poca::opengl::SsaoShader::SSAODebugDisplay::SSAO_COLOR));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_cboxSSAODisplaySSAOMap) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "currentDebugSSAO", poca::opengl::SsaoShader::SSAODebugDisplay::SSAO_SSAO_MAP));
		m_object->notifyAll("updateDisplay");
	}
}

void MainFilterWidget::updateDisplay(){
	m_object->executeCommand(&poca::core::CommandInfo(false, "updateDisplay"));
}

void MainFilterWidget::setColorBackgroundButton( const unsigned char _r, const unsigned char _g, const unsigned char _b )
{
	m_colorBackBtn->setStyleSheet( "background-color: rgb(" + QString::number( ( int )_r ) + ", " + QString::number( ( int )_g ) + ", " + QString::number( ( int )_b ) + ");"
		"border-style: outset;"
		"border-width: 2px;"
		"border-radius: 5px;"
		"border-color: black;"
		"font: 12px;"
		"min-width: 5em;"
		"padding: 3px;"
		);
}

void MainFilterWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
	if (_ci->nameCommand == "setImageIndex") {
		unsigned int val = _ci->getParameter<unsigned int>("id");
		poca::core::NbObjects = val;
		this->setFirstIndexObjVariable(val);
	}
	else if (_obj != NULL) {
		m_object = _obj;
		m_object->executeCommand(_ci);
		m_object->notifyAll("updateDisplay");
	}
}

void MainFilterWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo & _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	if (obj == NULL) return;
	m_object = obj;
	poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*> (_subject);
	if (comObj == NULL) return;
	if (_aspect == "LoadObjCharacteristicsAllWidgets"){
		QString nameDataset("Name: ");
		nameDataset.append(obj->getDir().c_str());
		m_nameDatasetLbl->setText(nameDataset);
		m_lineEditWidthData->setText(QString::number(obj->getWidth()));
		m_lineEditHeightData->setText(QString::number(obj->getHeight()));
		m_idDatasetLbl->setText("Id: " + QString::number(obj->currentInternalId()));
		
		if (comObj->hasParameter("pointSizeGL")) {
			m_sizePointSpn->blockSignals(true);
			this->setPointSize(comObj->getParameter<unsigned int>("pointSizeGL"));
			m_sizePointSpn->blockSignals(false);
		}
		if (comObj->hasParameter("smoothPoint"))
			this->setPointSmooth(comObj->getParameter<bool>("smoothPoint"));
		if (comObj->hasParameter("lineWidthGL"))
			this->setLineWidth(comObj->getParameter<uint32_t>("lineWidthGL"));
		if (comObj->hasParameter("smoothLine"))
			this->setLineSmooth(comObj->getParameter<bool>("smoothLine"));
		if (comObj->hasParameter("colorBakground")) {
			std::array <unsigned char, 4> rgba = comObj->getParameter< std::array <unsigned char, 4>>("colorBakground");
			this->setColorBackgroundButton(rgba[0], rgba[1], rgba[2]);
		}
		if (comObj->hasParameter("antialias"))
			m_antialiasCB->setChecked(comObj->getParameter<bool>("antialias"));
		if (comObj->hasParameter("cullFace"))
			m_cullFaceCB->setChecked(comObj->getParameter<bool>("cullFace"));
		if (comObj->hasParameter("clip"))
			m_clipCB->setChecked(comObj->getParameter<bool>("clip"));
		if (comObj->hasParameter("fillPolygon"))
			m_fillPolygonCB->setChecked(comObj->getParameter<bool>("fillPolygon"));

		//SSAO
		if (comObj->hasParameter("useSSAO"))
			m_cboxUseSSAO->setChecked(comObj->getParameter<bool>("useSSAO"));
		if (comObj->hasParameter("useSilhouetteSSAO"))
			m_cboxSSAOSilhouette->setChecked(comObj->getParameter<bool>("useSilhouetteSSAO"));
		if (comObj->hasParameter("useDebugSSAO"))
			m_cboxSSAOUseDebug->setChecked(comObj->getParameter<bool>("useDebugSSAO"));
		if (comObj->hasParameter("radiusSSAO")) {
			float radius = comObj->getParameter<float>("radiusSSAO");
			m_leditRadiusSSAO->setText(QString::number(radius));
			m_sliderRadiusSSAO->blockSignals(true);
			setRadiusSSAO(radius);
			m_sliderRadiusSSAO->blockSignals(false);
		}
		if (comObj->hasParameter("strengthSSAO")) {
			float strength = comObj->getParameter<float>("strengthSSAO");
			m_leditStrenghtSSAO->setText(QString::number(strength));
			m_sliderStrengthSSAO->blockSignals(true);
			m_sliderStrengthSSAO->setValue(strength / 0.05f);
			m_sliderStrengthSSAO->blockSignals(false);
		}

	}
	SMLMObject * sobj = dynamic_cast < SMLMObject * > ( _subject );
	if (sobj){
		if (_aspect == "LoadObjCharacteristicsAllWidgets"){
			m_dockGeneral->setVisible(true);
		}
	}
	else{
		m_dockGeneral->setVisible(true);
	}
}

void MainFilterWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo * _ci)
{
	this->performAction(_wobj, _ci);
}

