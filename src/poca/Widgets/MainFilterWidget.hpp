/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MainFilterWidget.hpp
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

#ifndef MainFilterWidget_h__
#define MainFilterWidget_h__

#include <QtWidgets/QWidget>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>

#include "DesignPatterns/Observer.hpp"
#include "DesignPatterns/MediatorWObjectFWidget.hpp"
#include "General/Command.hpp"
#include <OpenGL/Camera.hpp>

class QPlainTextEdit;
class QDockWidget;
class QLabel;

class MainFilterWidget: public QWidget, public poca::core::ObserverForMediator{
	Q_OBJECT

public:
	enum SelectionType{ TriangleSelectionFlag = 0x01, PolygonSelectionFlag = 0x02, ROISelectionFlag = 0x04, QuadTreeSelectionFlag = 0x08 };

	MainFilterWidget(poca::core::MediatorWObjectFWidget *, QWidget* = 0, Qt::WindowFlags = 0 );
	~MainFilterWidget();
	
	void setCurrentZoom(const double);
	void setZoomFactor(const double);

	inline void setFirstIndexObj( const unsigned int _val ){m_indexFirstImageEdit->setText( QString::number( _val ) );}
	inline bool isPointSmooth() const {return m_smoothPointCB->isChecked();}
	inline void setPointSmooth( const bool _val ){m_smoothPointCB->setChecked( _val );}
	inline bool isLineSmooth() const {return m_smoothLineCB->isChecked();}
	inline void setLineSmooth( const bool _val ){m_smoothLineCB->setChecked( _val );}
	inline int pointSize() const {return m_sizePointSpn->value();}
	inline void setPointSize( const int _val ){m_sizePointSpn->setValue( _val );}
	inline int lineWidth() const {return m_widthLineSpn->value();}
	inline float fontSize() const { return m_fontSizeSpn->value(); }
	inline void setLineWidth( const int _val ){m_widthLineSpn->setValue( _val );}
	inline int radiusSSAO() const { return m_sliderRadiusSSAO->value(); }
	inline int strengthSSAO() const { return m_sliderStrengthSSAO->value(); }
	inline void setRadiusSSAO(const int _val) { m_sliderRadiusSSAO->setValue(_val); }
	unsigned int getFirstIndexObj() const {return m_firstIndexObj;}
	inline void setFirstIndexObjVariable( const unsigned int _val ){m_firstIndexObj = _val;}

	void setColorBackgroundButton( const unsigned char, const unsigned char, const unsigned char );

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo & );
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

	inline void noChangingFirstImageIndexAnymore(){m_groupFirstImage->setVisible( false );}

	inline bool isViewCameraChecked() const { return m_cboxViewCamera->isChecked(); }
	inline bool isRotationCameraChecked() const { return m_cboxRotationCamera->isChecked(); }
	inline bool isTranslationCameraChecked() const { return m_cboxTranslationCamera->isChecked(); }
	inline bool isZoomCameraChecked() const { return m_cboxZoomCamera->isChecked(); }
	inline bool isCropCameraChecked() const { return m_cboxCropCamera->isChecked(); }

	inline void setCurrentCamera(poca::opengl::Camera* _cur) { m_currentCamera = _cur; }

protected slots:
	void actionNeeded();
	void actionNeeded( int );
	void actionNeeded( bool );
	void updateDisplay();

signals:
	void savePosition(QString);
	void loadPosition(QString);
	void getCurrentCamera();

protected:
	//General
	QCheckBox * m_smoothPointCB, * m_smoothLineCB, * m_antialiasCB, * m_cullFaceCB, * m_clipCB, * m_fillPolygonCB;
	QLabel * m_sizePointLbl, * m_widthLineLbl, * m_backColorLbl, * m_fontSizeLbl;
	QSpinBox * m_sizePointSpn, * m_widthLineSpn, * m_fontSizeSpn;
	QPushButton* m_colorBackBtn, * m_listCommandsBtn;

	//Infos
	QLabel * m_nameDatasetLbl, * m_idDatasetLbl;
	QLineEdit * m_lineEditWidthData, * m_lineEditHeightData;

	//Grid infos
	QCheckBox* m_cboxNbGrid, * m_cboxLengthGrid, * m_cboxSameAllDimGrid;
	QLineEdit* m_leditDimXNb, * m_leditDimYNb, * m_leditDimZNb;
	QLineEdit* m_leditDimXLength, * m_leditDimYLength, * m_leditDimZLength;

	/************************************************************************/
	/* For SSAO                                                             */
	/************************************************************************/
	QCheckBox* m_cboxUseSSAO, * m_cboxSSAOSilhouette, * m_cboxSSAODisplayPos, * m_cboxSSAODisplayNormal, * m_cboxSSAODisplayColor, * m_cboxSSAODisplaySSAOMap, * m_cboxSSAOUseDebug;
	QLabel* m_lblRadiusSSAO, * m_lblStrengthSSAO;
	QLineEdit* m_leditRadiusSSAO, * m_leditStrenghtSSAO;
	QSlider* m_sliderRadiusSSAO, * m_sliderStrengthSSAO;

	//For camera position
	QDockWidget* m_dockCameraPosition;
	QPushButton* m_savePositionBtn, * m_loadPositionBtn;
	QCheckBox* m_cboxViewCamera, * m_cboxRotationCamera, * m_cboxTranslationCamera, * m_cboxZoomCamera, * m_cboxCropCamera;

	QGroupBox * m_groupFirstImage;
	QDockWidget * m_dockGeneral, * m_dockInfoDataset, * m_dockGrid, * m_dockSSAO;

	QLineEdit * m_indexFirstImageEdit;
	QPushButton * m_setFirstImageIdBtn;

	QWidget* m_emptyWidget;

	unsigned int m_firstIndexObj;

	poca::core::MyObjectInterface* m_object;
	poca::opengl::Camera* m_currentCamera;

	poca::core::MediatorWObjectFWidget * m_mediator;
};

#endif // MainFilterWidget_h__

