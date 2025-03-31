/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocTesselerWidget.hpp
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

#ifndef VoronoiDiagramWidget_h__
#define VoronoiDiagramWidget_h__

#include <QtWidgets/QTabWidget>
#include <QtWidgets/QLineEdit>

#include <DesignPatterns/Observer.hpp>
#include <General/Command.hpp>
#include <General/Palette.hpp>
#include <Plot/ScatterplotGL.hpp>

#include "MyTableWidget.hpp"

class QPushButton;
class QCheckBox;
class QGroupBox;
class QSlider;

//! [0]
class ColocTesselerWidget : public QTabWidget, public poca::core::ObserverForMediator {
	Q_OBJECT

public:
	ColocTesselerWidget(poca::core::MediatorWObjectFWidgetInterface*, QWidget* = 0);
	~ColocTesselerWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&);
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

	const float getDensityFactor(size_t, bool*) const;
	inline void setParentTab(QTabWidget* _tab) { m_parentTab = _tab; }

protected:

protected slots:
	void actionNeeded();
	void actionNeeded(bool);
	void actionNeeded(int);
	void actionNeeded(const QString&);

protected:
	QTabWidget* m_parentTab;
	poca::core::MediatorWObjectFWidgetInterface* m_mediator;

	QWidget* m_emptyWidget;
	QGroupBox* m_actionsGBox;
	QPushButton* m_displayButton, * m_applyFactorButton, * m_saveButton;
	QCheckBox* m_correctionCbox, * m_inROIsCbox;
	QLineEdit* m_factorDensityEdit[2];
	QLineEdit* m_minRadiusEdit, * m_maxRadiusEdit, * m_currentRadiusEdit, * m_intensityEdit;
	QSlider* m_radiusSlider, * m_intensitySlider;
	QCheckBox* m_logHeatMapCBox;
	std::vector <std::pair<QPushButton*, std::string>> m_lutHeatmapButtons;
	MyTableWidget* m_tableColoc;

	poca::plot::ScatterplotGL* m_scatterGL;
	
	poca::core::MyObjectInterface* m_object;
};

//! [0]
#endif

